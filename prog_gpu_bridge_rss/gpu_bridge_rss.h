/*
 * gpu_bridge_rss.h — strutture, costanti e dichiarazioni per il GPU L2 Bridge
 * a N porte con RSS multi-coda (v2).
 *
 * DIFFERENZA RISPETTO A prog_gpu_bridge (v1):
 *
 *   v1: 1 sola RXQ per porta, 1 blocco CUDA da 32 thread che serve TUTTE
 *       le porte in round-robin (loop seriale "for src in 0..n_ports").
 *
 *   v2: N_QUEUES_PER_PORT RXQ per porta (bilanciate via RSS hardware sulla
 *       NIC), e un blocco CUDA da 32 thread DEDICATO per ogni coppia
 *       (porta, coda). Ogni blocco è "self-contained": possiede la propria
 *       RXQ e un set di TXQ private (una per porta di destinazione), quindi
 *       non c'è MAI contesa tra blocchi diversi sul ring WQE di una TXQ —
 *       zero sincronizzazione cross-block richiesta.
 *
 *   Il prezzo di questo design (nessuna sync cross-block) è un numero
 *   maggiore di code hardware: ogni porta fisica ospita
 *     N_QUEUES_PER_PORT RX queue (le proprie)
 *     N_QUEUES_PER_PORT * (n_ports - 1) TX queue (una per ogni blocco di
 *       ogni ALTRA porta che potrebbe forwardare verso di essa)
 *   Vedi check_queue_budget() in gpu_bridge_rss.c per il controllo contro
 *   il limite hardware (tipicamente 63 code per tipo per porta su BF2/BF3).
 *
 * Uso:
 *   sudo ip netns exec bf2 ./gpu_bridge_rss \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0
 */

#ifndef GPU_BRIDGE_RSS_H
#define GPU_BRIDGE_RSS_H

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
 * COSTANTI — NUMERO DI PORTE E CODE
 * =========================================================================
 *
 * MAX_N_PORTS: numero massimo di porte supportate a compile time.
 *   Il numero effettivo di porte (n_ports) viene passato a runtime (-n).
 *
 * N_QUEUES_PER_PORT: numero di code RSS per porta, FISSO a compile time
 *   (scelta esplicita: niente flag CLI, per tenere il grid layout semplice
 *   e prevedibile). La scheda BF2/BF3 supporta fino a 63 code; qui usiamo
 *   un valore "nell'ordine della decina" per sfruttare bene il parallelismo
 *   della GPU senza esplodere il numero di code hardware (vedi
 *   check_queue_budget() per il conto preciso in funzione di n_ports).
 */
#define MAX_N_PORTS        8
#define N_QUEUES_PER_PORT  16

/* =========================================================================
 * COSTANTI — CODA DI RICEZIONE (RXQ)
 * =========================================================================
 *
 * MAX_PKT_NUM: slot nel ring buffer ciclico (CYCLIC RXQ) PER CODA.
 *   16384 slot × MAX_PKT_SIZE byte = 32 MB per coda.
 *   ATTENZIONE VRAM: memoria RX totale ≈ n_ports * N_QUEUES_PER_PORT * 32 MB.
 *   Con n_ports=2, N_QUEUES_PER_PORT=16: 2*16*32MB = 1 GiB. Su GPU con poca
 *   VRAM o con molte porte, ridurre MAX_PKT_NUM o N_QUEUES_PER_PORT.
 *
 * MAX_PKT_SIZE: dimensione massima per pacchetto (Ethernet 1500 + overhead).
 *
 * MAX_RX_TIMEOUT_NS: timeout della recv nel kernel.
 *
 * MAX_RX_NUM_PKTS: max pacchetti per singola recv call (per blocco/coda).
 *   32 thread × 64 = 2048 pacchetti elaborabili in parallelo per blocco.
 */
#define MAX_PKT_NUM       16384
#define MAX_PKT_SIZE      2048
#define MAX_RX_TIMEOUT_NS 500000
#define MAX_RX_NUM_PKTS   2048

/* =========================================================================
 * COSTANTI — CODA DI TRASMISSIONE (TXQ)
 * =========================================================================
 * MAX_SQ_DESCR_NUM: WQE per TXQ. Ogni TXQ v2 è privata di UN blocco verso
 * UNA porta di destinazione, quindi vede una frazione del traffico rispetto
 * a v1: 1024 resta ampiamente conservativo.
 */
#define MAX_SQ_DESCR_NUM  2048

/* =========================================================================
 * COSTANTI — DOCA FLOW
 * =========================================================================
 */
#define FLOW_NB_COUNTERS  512

/*
 * RSS_HASH_FIELDS: campi usati dalla NIC per l'hash Toeplitz che sceglie
 * la coda RX. Copre il traffico IPv4/UDP (quello del profilo di test
 * gpu_bridge_rss_stress.py).
 *
 * NON combinare qui più di un flag L3 (IPV4/IPV6) o più di un flag L4
 * (UDP/TCP) insieme: in TUTTI i sample e le applicazioni ufficiali DOCA che
 * usano RSS multi-coda (grep su .../doca/samples e .../doca/applications),
 * senza eccezioni, outer_flags è sempre "un solo L3 + un solo L4" (es.
 * IPV4|UDP oppure IPV4|TCP). Combinare IPV4|IPV6|UDP|TCP insieme su
 * un'unica pipe (come in una versione precedente di questo file) fa
 * fallire doca_flow_pipe_create con DOCA_ERROR_INVALID_VALUE. Per bilanciare
 * anche TCP e/o IPv6, serve una pipe RSS separata per combinazione,
 * incatenata dietro una pipe di classificazione — vedi il sample ufficiale
 * applications/gpu_packet_processing/config_queues/flow.c per il pattern.
 *
 * Il traffico L2 puro (ARP, ecc. — nessun header IP) NON ha campi su cui
 * fare hash: DOCA Flow/la NIC lo instrada su una coda fissa (tipicamente
 * la 0). È un limite hardware, non del nostro codice: la struttura
 * doca_rss_type non espone alcun campo di hash sui MAC L2.
 */
#define RSS_HASH_FIELDS \
    (DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP)

/* =========================================================================
 * COSTANTI — CUDA KERNEL
 * =========================================================================
 * BRIDGE_BLOCK_THREADS = 32 = 1 warp NVIDIA, come in v1.
 * Un blocco per ogni coppia (porta, coda): vedi bridge_queue_ctx sotto.
 */
#define BRIDGE_BLOCK_THREADS 32

/* =========================================================================
 * COSTANTI — MAC TABLE (FIB), condivisa da TUTTI i blocchi/porte/code
 * =========================================================================
 */
#define MAC_TABLE_SIZE       4096
#define MAC_TABLE_PROBE_MAX  256
#define MAC_ENTRY_VALID_BIT  ((uint64_t)1 << 63)
#define MAC_ENTRY_PORT_SHIFT 48
#define MAC_ENTRY_PORT_MASK  0x00FF000000000000ULL
#define MAC_48BIT_MASK       0x0000FFFFFFFFFFFFULL

/* =========================================================================
 * DIAGNOSTICA — MAC FLAP DETECTION (invariata rispetto a v1)
 * =========================================================================
 */
#define FLAP_RING_SIZE 64

struct mac_flap_record {
    uint64_t mac48;
    uint32_t old_port;
    uint32_t new_port;
    uint64_t seq;
};

/* =========================================================================
 * MACRO — ALLINEAMENTO MEMORIA
 * =========================================================================
 */
#define ALIGN_UP(sz, align) (((sz) + (align) - 1) / (align) * (align))

/* =========================================================================
 * STRUTTURE LATO CPU — RISORSE PER CODA
 * =========================================================================
 */

/* Una RXQ GPU (buffer ciclico + mmap cross-port), una per (porta, coda). */
struct bridge_rxq_slot {
    struct doca_ctx         *rxq_ctx;
    struct doca_eth_rxq     *rxq_cpu;
    struct doca_gpu_eth_rxq *rxq_gpu;
    struct doca_mmap        *rxq_mmap;
    void                    *rxq_buf;
    int                      rxq_dmabuf_fd;
    /* mkey_for_port[d] = mkey del buffer di QUESTA rxq, valido per la NIC
     * della porta d (autorizza la NIC d a fare DMA READ da qui). */
    uint32_t                 mkey_for_port[MAX_N_PORTS];
};

/* Una TXQ GPU, privata di un blocco (porta sorgente, coda) verso UNA
 * porta di destinazione. Nessun buffer dati proprio: i WQE puntano al
 * buffer della rxq_slot sorgente (zero-copy cross-port). */
struct bridge_txq_slot {
    struct doca_ctx         *txq_ctx;
    struct doca_eth_txq     *txq_cpu;
    struct doca_gpu_eth_txq *txq_gpu;
};

/* Una porta fisica del bridge. */
struct bridge_port {
    struct doca_dev *ddev;

    struct doca_flow_port       *flow_port;
    struct doca_flow_pipe       *root_pipe;
    struct doca_flow_pipe_entry *root_entry;

    /* Le N_QUEUES_PER_PORT code RX di questa porta (bilanciate da RSS). */
    struct bridge_rxq_slot rxq[N_QUEUES_PER_PORT];

    /* Prossimo queue_id TX libero su QUESTA NIC (namespace TX separato
     * da quello RX). Incrementato ogni volta che un blocco di un'ALTRA
     * porta crea qui la propria TXQ privata verso questa porta. */
    uint16_t next_txq_id;
};

/* =========================================================================
 * STRUTTURA GPU-RESIDENT — bridge_queue_ctx
 * =========================================================================
 * Un'istanza per ogni blocco CUDA (una per coppia (porta, coda)).
 * Vive in un array in memoria GPU puntato da bridge_kernel_params.queues;
 * ogni blocco legge SOLO la propria istanza (kp.queues[blockIdx.x]).
 *
 * Tenerla fuori dai parametri kernel "by value" è necessario: con
 * n_ports*N_QUEUES_PER_PORT*MAX_N_PORTS puntatori TXQ, la struct
 * supererebbe facilmente il limite di kernel parameter memory (~4 KB)
 * se passata per valore come in v1.
 */
struct bridge_queue_ctx {
    int port;   /* porta sorgente servita da questo blocco (0..n_ports-1) */
    int queue;  /* indice coda entro quella porta (0..N_QUEUES_PER_PORT-1) */

    struct doca_gpu_eth_rxq *rxq_gpu;

    /* txq_gpu[d]: TXQ privata di QUESTO blocco verso la porta d.
     * NULL per d == port (non si forwarda mai verso la porta di ingresso). */
    struct doca_gpu_eth_txq *txq_gpu[MAX_N_PORTS];

    /* rxq_mkey_for_dst[d]: mkey del buffer rxq di QUESTO blocco, valido
     * per la NIC della porta d (zero-copy cross-port). */
    uint32_t rxq_mkey_for_dst[MAX_N_PORTS];
};

/* =========================================================================
 * STRUTTURA bridge_kernel_params
 * =========================================================================
 * Passata per valore al kernel (piccola: solo scalari + puntatori GPU).
 */
struct bridge_kernel_params {
    int n_ports;
    int n_queues;                    /* = N_QUEUES_PER_PORT */
    struct bridge_queue_ctx *queues; /* array GPU, n_ports*n_queues elementi */

    uint64_t *mac_table;
    uint32_t *exit_cond;
    uint64_t *fwd_count;             /* contatore aggregato, atomicAdd da ogni blocco */

    /* ── Diagnostica live (CPU_GPU), aggregata con atomicAdd da tutti i
     * blocchi una volta per ogni batch ricevuto (non per pacchetto). ──── */
    uint64_t *rx_pkt_total;   /* [n_ports] */
    /* rx_pkt_per_queue[p*n_queues+q]: pacchetti ricevuti dalla singola coda
     * (p,q). Serve SOLO a verificare che l'RSS hardware stia davvero
     * distribuendo il traffico su tutte le code (le RXQ DOCA GPUNetIO sono
     * code raw DevX: non compaiono in "ethtool -S", questo è l'unico modo
     * di osservarle). Indice = blockIdx.x, stesso ordine di kp.queues. */
    uint64_t *rx_pkt_per_queue; /* [n_ports * n_queues] */
    uint64_t *flood_count;
    uint64_t *unicast_count;
    uint64_t *drop_count;
    struct mac_flap_record *flap_ring;  /* [FLAP_RING_SIZE] */
    uint32_t *flap_ring_head;
};

#ifdef __cplusplus
extern "C" {
#endif

doca_error_t kernel_launch_bridge(cudaStream_t stream,
                                   struct bridge_kernel_params *kp);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BRIDGE_RSS_H */
