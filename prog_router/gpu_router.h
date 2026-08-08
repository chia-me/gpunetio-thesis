/*
 * gpu_router.h — strutture, costanti e dichiarazioni per il GPU L3 Router a N porte.
 *
 * COS'È QUESTO PROGRAMMA (in breve):
 *   Un router IPv4 il cui *data plane* (il ciclo "ricevi, decidi, riscrivi,
 *   trasmetti" che gira per ogni singolo pacchetto) è interamente eseguito
 *   dalla GPU, mentre la CPU si occupa solo del *control plane* (leggere la
 *   tabella di routing da file, aprire i device DOCA, configurare le code).
 *
 *   A differenza del bridge (prog_gpu_bridge, L2, decide in base al MAC),
 *   questo router lavora al Livello 3 (IP):
 *     - guarda l'IP di destinazione del pacchetto, non il MAC
 *     - fa un Longest Prefix Match (LPM) su una tabella di rotte statiche
 *     - decrementa la TTL e ricalcola il checksum IPv4 (obbligatorio: un
 *       router, per definizione dello standard IP, altera l'header ad ogni
 *       hop; un bridge L2 non lo fa mai)
 *     - riscrive COMPLETAMENTE l'header Ethernet (MAC sorgente = MAC della
 *       propria interfaccia di uscita, MAC destinazione = next-hop) invece
 *       di limitarsi a inoltrare il frame L2 originale
 *
 * RIB vs FIB (terminologia, per chiarezza — vedi anche i commenti nel .c):
 *   - RIB (Routing Information Base): la tabella "grezza" delle rotte così
 *     come configurate/apprese (qui: righe lette dal file di rotte). È la
 *     vista del control plane, non ottimizzata per la ricerca ad alta
 *     velocità.
 *   - FIB (Forwarding Information Base): la tabella *compilata* e ordinata,
 *     pronta per la ricerca sul data plane. In questo programma: lo stesso
 *     array della RIB ma ordinato per lunghezza di prefisso decrescente,
 *     copiato in memoria GPU. Un vero router (es. con BGP/OSPF) avrebbe
 *     una RIB dinamica enorme e una FIB ricompilata periodicamente; qui,
 *     senza protocolli di routing dinamici, RIB e FIB hanno lo stesso
 *     contenuto — cambia solo la struttura dati e dove risiedono in memoria.
 *
 * Uso:
 *   sudo ip netns exec bf2 ./gpu_router \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0 -r routes.txt
 */

#ifndef GPU_ROUTER_H
#define GPU_ROUTER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_eth_def.h>
#include <doca_eth_rxq.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_flow.h>

/* =========================================================================
 * COSTANTI — NUMERO DI PORTE
 * =========================================================================
 * Identico al bridge: il router è generico su N porte, il massimo a
 * compile-time è MAX_N_PORTS, il numero effettivo (n_ports) è a runtime.
 */
#define MAX_N_PORTS 8

/* =========================================================================
 * COSTANTI — CODA DI RICEZIONE (RXQ)
 * =========================================================================
 * Stessi valori del bridge: sono dimensionamenti del datapath GPUNetIO,
 * non hanno nulla a che vedere con L2 vs L3.
 */
#define MAX_PKT_NUM       16384
#define MAX_PKT_SIZE      2048
#define MAX_RX_TIMEOUT_NS 500000
#define MAX_RX_NUM_PKTS   2048

/* =========================================================================
 * COSTANTI — CODA DI TRASMISSIONE (TXQ)
 * =========================================================================
 * NOTA IMPORTANTE rispetto al bridge: un router IP unicast NON floода mai
 * (nessun broadcast/multicast gestito in questa implementazione — vedi
 * i commenti in gpu_router_kernel.cu). Ogni pacchetto ricevuto genera al
 * più UN solo WQE in uscita (non N-1 come nel flooding del bridge), quindi
 * un numero di descrittori inferiore è già sufficiente. Manteniamo comunque
 * lo stesso valore del bridge per margine operativo e uniformità.
 */
#define MAX_SQ_DESCR_NUM  1024

/* =========================================================================
 * COSTANTI — DOCA FLOW
 * =========================================================================
 */
#define FLOW_NB_COUNTERS  512

/* =========================================================================
 * COSTANTI — CUDA KERNEL
 * =========================================================================
 * BRIDGE_BLOCK_THREADS = 32 = 1 warp. Vedi gpu_bridge.h per la spiegazione
 * completa del perché EXEC_SCOPE_BLOCK richiede l'intero warp: vale
 * identica qui, la ricezione (doca_gpu_dev_eth_rxq_recv) è la stessa API.
 */
#define ROUTER_BLOCK_THREADS 32

/* =========================================================================
 * COSTANTI — TABELLA DI ROUTING (RIB/FIB)
 * =========================================================================
 * MAX_ROUTES: numero massimo di rotte statiche caricabili da file.
 *   1024 è ampiamente sufficiente per una topologia di laboratorio/test;
 *   con un numero di rotte così piccolo una scansione lineare (vedi
 *   fib_lookup in gpu_router_kernel.cu) è più che adeguata in termini di
 *   prestazioni ed è enormemente più semplice da implementare, verificare
 *   e leggere di una trie binaria (Patricia trie / DIR-24-8) usata dai
 *   router hardware reali con centinaia di migliaia di rotte BGP.
 *   Questo è un compromesso esplicito: didattico e corretto, non pensato
 *   per una full Internet routing table.
 */
#define MAX_ROUTES 1024

/* =========================================================================
 * MACRO — ALLINEAMENTO MEMORIA
 * =========================================================================
 */
#define ALIGN_UP(sz, align) (((sz) + (align) - 1) / (align) * (align))

/* =========================================================================
 * MACRO — BYTE SWAP (network byte order <-> host byte order)
 * =========================================================================
 * I campi multi-byte di un pacchetto (lunghezza IP, porte TCP/UDP, ecc.)
 * viaggiano in rete in "network byte order" (big-endian). Le CPU x86/ARM
 * di questo sistema sono little-endian: per confrontare o stampare quei
 * campi come numeri "giusti" serve invertire i byte. Stesse macro usate
 * nel sample ufficiale NVIDIA gpu_packet_processing (defines.h).
 *
 * NOTA: per l'IP di destinazione NON serve alcun BYTE_SWAP (vedi la nota
 * "NESSUN BYTE-SWAP IN HOT PATH" più sotto e in gpu_router_kernel.cu):
 * lo confrontiamo con le rotte tenendo entrambi in network byte order.
 * Usiamo BYTE_SWAP16 solo dove serve leggere un valore come numero
 * (es. total_length nei log/diagnostica).
 */
#define BYTE_SWAP16(v) ((((uint16_t)(v) & UINT16_C(0x00ff)) << 8) | (((uint16_t)(v) & UINT16_C(0xff00)) >> 8))

/* =========================================================================
 * STRUTTURE PACCHETTO — HEADER ETHERNET E IPv4
 * =========================================================================
 * Layout identico (nomi e semantica) a quello usato nel sample ufficiale
 * NVIDIA DOCA "gpu_packet_processing" (packets.h), per conformità con le
 * convenzioni DOCA GPUNetIO: struct "packed" sovrapposta direttamente sul
 * buffer di ricezione (nessuna copia), bitfield per version/ihl che sulle
 * architetture little-endian (x86_64/aarch64, entrambe usate da host e
 * BlueField) rispecchiano l'ordine reale sul wire (ihl nel nibble basso,
 * version nel nibble alto del primo byte).
 */
#define ETHER_ADDR_LEN 6

struct ether_hdr {
    uint8_t  d_addr_bytes[ETHER_ADDR_LEN]; /* MAC destinazione, ordine di trasmissione */
    uint8_t  s_addr_bytes[ETHER_ADDR_LEN]; /* MAC sorgente, ordine di trasmissione */
    uint16_t ether_type;                   /* EtherType, network byte order */
} __attribute__((__packed__));

struct ipv4_hdr {
    union {
        uint8_t version_ihl;
        struct {
            uint8_t ihl     : 4; /* Internet Header Length, in parole da 32 bit */
            uint8_t version : 4; /* sempre 4 per IPv4 */
        };
    };
    uint8_t  type_of_service;
    uint16_t total_length;   /* network byte order */
    uint16_t packet_id;      /* network byte order */
    uint16_t fragment_offset;/* network byte order */
    uint8_t  time_to_live;
    uint8_t  next_proto_id;
    uint16_t hdr_checksum;   /* network byte order */
    uint32_t src_addr;       /* network byte order */
    uint32_t dst_addr;       /* network byte order */
} __attribute__((__packed__));

struct eth_ip_hdr {
    struct ether_hdr l2_hdr;
    struct ipv4_hdr  l3_hdr;
} __attribute__((__packed__));

/* =========================================================================
 * STRUTTURA route_entry — UNA RIGA DELLA RIB/FIB
 * =========================================================================
 * network / mask:
 *   Entrambi in NETWORK BYTE ORDER, esattamente come appaiono nel campo
 *   dst_addr di un pacchetto IPv4 in arrivo. Questo è deliberato: confrontare
 *   "dst_addr & mask == network" direttamente sui byte-as-received evita
 *   ogni operazione di byte-swap nel percorso critico (hot path) del kernel
 *   CUDA — lo swap va fatto (via inet_pton/htonl) una sola volta, in fase
 *   di caricamento del file di rotte sulla CPU, non per ogni pacchetto.
 *   Vedi build_mask_be() e load_routes_file() in gpu_router.c.
 *
 * prefix_len:
 *   Usato SOLO lato CPU per ordinare la FIB dal prefisso più lungo al più
 *   corto (vedi route_cmp_by_prefixlen_desc). Non serve al kernel: la
 *   maschera già codifica la stessa informazione.
 *
 * egress_port:
 *   Indice di porta (0..n_ports-1) su cui inoltrare i pacchetti che
 *   fanno match su questa rotta.
 *
 * next_hop_mac:
 *   MAC address da scrivere come MAC DESTINAZIONE del frame in uscita
 *   (il "prossimo salto" L2, sia esso il router successivo o l'host finale
 *   se la rotta è per una rete direttamente connessa). Questo router non
 *   implementa ARP: il next-hop MAC è statico, fornito nel file di rotte
 *   (concettualmente equivalente ad avere una voce ARP permanente per
 *   ogni rotta).
 */
struct route_entry {
    uint32_t network;
    uint32_t mask;
    uint8_t  prefix_len;
    uint8_t  egress_port;
    uint8_t  next_hop_mac[ETHER_ADDR_LEN];
};

/* =========================================================================
 * STRUTTURA router_port
 * =========================================================================
 * Rappresenta una porta fisica del router. Il main alloca:
 *   struct router_port ports[n_ports] con n_ports <= MAX_N_PORTS.
 *
 * Identica nella parte RXQ/TXQ/Flow al bridge_port del bridge: il datapath
 * GPUNetIO (codice di RICEZIONE dalla NIC e TRASMISSIONE verso la NIC) non
 * dipende dal fatto che si stia facendo bridging L2 o routing L3 — cambia
 * solo cosa il kernel CUDA fa CON i dati una volta ricevuti.
 */
struct router_port {
    struct doca_dev *ddev;

    /* Il MAC address di QUESTA interfaccia, letto dal device DOCA.
     * Usato dal kernel come MAC SORGENTE quando un pacchetto viene
     * ritrasmesso da questa porta (comportamento standard di un router:
     * ad ogni hop il MAC sorgente diventa quello dell'interfaccia
     * d'uscita, non quello del mittente originale). */
    uint8_t own_mac[ETHER_ADDR_LEN];

    /* ── Coda di ricezione (RXQ GPU CYCLIC) ─────────────────────────── */
    struct doca_ctx         *rxq_ctx;
    struct doca_eth_rxq     *rxq_cpu;
    struct doca_gpu_eth_rxq *rxq_gpu;
    struct doca_mmap        *rxq_mmap;
    void                    *rxq_buf;
    int                      rxq_dmabuf_fd;
    uint32_t                 rxq_mkey_for_port[MAX_N_PORTS]; /* [q] = mkey per NIC porta q */

    /* ── Coda di trasmissione (TXQ GPU) ─────────────────────────────── */
    struct doca_ctx         *txq_ctx;
    struct doca_eth_txq     *txq_cpu;
    struct doca_gpu_eth_txq *txq_gpu;

    /* ── DOCA Flow: BASIC ROOT pipe con match "solo IPv4" ────────────── */
    struct doca_flow_port       *flow_port;
    struct doca_flow_pipe       *root_pipe;
    struct doca_flow_pipe_entry *root_entry;
};

/* =========================================================================
 * STRUTTURA router_kernel_params
 * =========================================================================
 * Passata per valore al kernel CUDA (come in gpu_bridge.h: CUDA copia
 * l'intera struct in kernel parameter memory, <=4KB).
 *
 * fib / fib_size:
 *   Puntatore GPU alla FIB (array di route_entry) e numero di rotte valide.
 *   A differenza della mac_table del bridge, la FIB è READ-ONLY per tutta
 *   la vita del kernel: viene scritta una sola volta dalla CPU prima del
 *   lancio (routes.txt -> RIB -> sort -> FIB -> cudaMemcpy) e mai più
 *   modificata. Questo elimina completamente il bisogno di atomici o
 *   accessi volatili per la FIB (a differenza della mac_table del bridge,
 *   che viene scritta continuamente dal MAC learning).
 *
 * port_mac[p]:
 *   MAC address della porta p (= ports[p].own_mac), copiato qui per averlo
 *   disponibile ai thread GPU senza dereferenziare puntatori CPU.
 */
struct router_kernel_params {
    int      n_ports;
    struct doca_gpu_eth_rxq *rxq_gpu[MAX_N_PORTS];
    struct doca_gpu_eth_txq *txq_gpu[MAX_N_PORTS];
    uint32_t rxq_mkey_cross[MAX_N_PORTS][MAX_N_PORTS]; /* [src][dst] = mkey rxq[src] per NIC dst */
    uint8_t  port_mac[MAX_N_PORTS][ETHER_ADDR_LEN];

    struct route_entry *fib;
    uint32_t             fib_size;

    uint32_t *exit_cond;
    uint64_t *fwd_count;

    /* ── Diagnostica live (CPU_GPU, pubblicata una volta per ogni giro
     * completo su tutte le porte, con __threadfence_system) ──────────── */
    uint64_t *rx_pkt_total;      /* [n_ports]: pacchetti ricevuti realmente da RXQ */
    uint64_t *drop_no_route;     /* nessuna rotta compatibile (LPM miss) */
    uint64_t *drop_ttl_expired;  /* TTL sarebbe arrivata a 0: pacchetto scartato */
    uint64_t *drop_malformed;    /* IPv4 con IHL/versione non validi (difesa in profondità) */
};

/* =========================================================================
 * DICHIARAZIONE WRAPPER KERNEL
 * =========================================================================
 */
#ifdef __cplusplus
extern "C" {
#endif

doca_error_t kernel_launch_router(cudaStream_t stream,
                                   struct router_kernel_params *kp);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ROUTER_H */
