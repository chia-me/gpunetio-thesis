/*
 * gpu_bridge.h — strutture, costanti e dichiarazioni per il GPU L2 Bridge a N porte.
 *
 * Il bridge supporta da 2 a MAX_N_PORTS porte fisiche.
 * Il numero effettivo di porte è stabilito a runtime (-n flag sulla command line).
 *
 * Ogni pacchetto ricevuto su una porta viene elaborato dalla GPU A30X:
 *   - backward learning: src_mac → porta di ingresso nella FIB
 *   - FIB lookup: dst_mac → porta di uscita (unicast), -1 (flood) o drop
 *   - WQE filling parallelo con atomicAdd: ogni thread scrive il proprio WQE
 *   - zero-copy cross-port: i WQE puntano al buffer GPU della porta sorgente
 *
 * Uso:
 *   sudo ip netns exec bf2 ./gpu_bridge \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0
 */

#ifndef GPU_BRIDGE_H
#define GPU_BRIDGE_H

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
 *
 * MAX_N_PORTS: numero massimo di porte supportate a compile time.
 *   Dimensiona gli array statici in kernel params e bridge_port.
 *   Il numero effettivo di porte (n_ports) viene passato al kernel a runtime.
 *
 * 8 è sufficiente per BF2/BF3 con SR-IOV VF e porta di management.
 */
#define MAX_N_PORTS 8

/* =========================================================================
 * COSTANTI — CODA DI RICEZIONE (RXQ)
 * =========================================================================
 *
 * MAX_PKT_NUM: slot nel ring buffer ciclico (CYCLIC RXQ).
 *   La NIC DMA i pacchetti in questo buffer circolare in VRAM.
 *   16384 slot × MAX_PKT_SIZE byte = 32 MB per porta.
 *
 * MAX_PKT_SIZE: dimensione massima per pacchetto (Ethernet 1500 + overhead).
 *
 * MAX_RX_TIMEOUT_NS: timeout della recv nel kernel.
 *   500 µs: buon compromesso tra reattività al Ctrl+C e efficienza.
 *
 * MAX_RX_NUM_PKTS: max pacchetti per singola recv call.
 *   32 thread × 64 = 2048 pacchetti elaborabili in parallelo.
 */
#define MAX_PKT_NUM       16384
#define MAX_PKT_SIZE      2048
#define MAX_RX_TIMEOUT_NS 500000
#define MAX_RX_NUM_PKTS   2048

/* =========================================================================
 * COSTANTI — CODA DI TRASMISSIONE (TXQ)
 * =========================================================================
 *
 * MAX_SQ_DESCR_NUM: Work Queue Entry (WQE) nel Send Queue della TXQ.
 *   Con flooding a N porte, ogni pacchetto può generare N-1 WQE.
 *   1024 WQE è conservativo.
 */
#define MAX_SQ_DESCR_NUM  2048

/* POLL_NB_TIMEOUT_ITERS: numero massimo di controlli non bloccanti (NB) su
 * una singola CQE attesa prima di considerarla "mai arrivata" e abortire
 * in modo pulito invece di restare bloccati per sempre in poll_completion_at
 * (WAIT_FLAG_B). Valore scelto empiricamente, non calibrato su un tempo
 * reale preciso: se in pratica scatta durante condizioni di carico normali
 * (falsi positivi), va alzato; se il freeze reale richiede troppo tempo per
 * essere rilevato, va abbassato. */
#define POLL_NB_TIMEOUT_ITERS  20000000u

/* =========================================================================
 * COSTANTI — DOCA FLOW
 * =========================================================================
 */
#define FLOW_NB_COUNTERS  512

/* =========================================================================
 * COSTANTI — CUDA KERNEL
 * =========================================================================
 *
 * BRIDGE_BLOCK_THREADS = 32 = 1 warp NVIDIA.
 * Richiesto da EXEC_SCOPE_BLOCK: tutti i thread cooperano per la recv.
 * Il tracking del "last WQE" per il fixup NOTIFY (vedi gpu_bridge_kernel.cu)
 * usa atomicMax su una chiave (slot, pkt_idx) e non richiede alcuna
 * assunzione di lockstep, quindi è corretto anche su architetture con
 * Independent Thread Scheduling (Volta+).
 */
#define BRIDGE_BLOCK_THREADS 32

/* =========================================================================
 * COSTANTI — MAC TABLE (FIB)
 * =========================================================================
 *
 * Formato entry uint64_t:
 *   bit 63:    valid flag (1 = slot occupato, 0 = slot vuoto)
 *   bit 48-55: porta di uscita (0..255, supporta fino a 256 porte)
 *   bit 0-47:  MAC address a 48 bit
 *
 * Con MAX_N_PORTS = 8, i valori del campo porta sono 0-7.
 * I bit 56-62 sono riservati e sempre zero.
 *
 * MAC_TABLE_SIZE deve essere potenza di 2 per l'AND hash trick.
 * MAC_TABLE_PROBE_MAX: passi max di linear probing prima di rinunciare.
 *   Con tabella piena: mac_lookup ritorna -1 (flood), mac_learn ignora.
 */
#define MAC_TABLE_SIZE       4096
#define MAC_TABLE_PROBE_MAX  256
#define MAC_ENTRY_VALID_BIT  ((uint64_t)1 << 63)
#define MAC_ENTRY_PORT_SHIFT 48
#define MAC_ENTRY_PORT_MASK  0x00FF000000000000ULL  /* bit 48-55: porta (0-255) */
#define MAC_48BIT_MASK       0x0000FFFFFFFFFFFFULL  /* bit 0-47:  MAC address */

/* =========================================================================
 * DIAGNOSTICA — MAC FLAP DETECTION
 * =========================================================================
 * Un "flap" e' quando un MAC gia' presente nella FIB viene re-imparato
 * su una porta DIVERSA da quella registrata. In una topologia senza loop
 * fisico questo non dovrebbe mai succedere (un host sta sempre dietro
 * la stessa porta). Flap frequenti sono la prova classica di un loop L2
 * reale nella rete (il frame torna indietro e viene re-imparato al contrario).
 *
 * flap_ring_head e' un contatore atomico monotono: il suo valore intero
 * E' il totale dei flap avvenuti; (head % FLAP_RING_SIZE) e' lo slot dove
 * viene scritto il record più recente (ring buffer, i piu' vecchi vengono
 * sovrascritti).
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
 * STRUTTURA bridge_port
 * =========================================================================
 * Rappresenta una porta fisica del bridge.
 * Il main alloca: struct bridge_port ports[n_ports] con n_ports ≤ MAX_N_PORTS.
 *
 * Cross-port mkey (generalizzato a N porte):
 *   rxq_mkey_for_port[q] = mkey del buffer GPU di questa porta,
 *                           valido per la NIC della porta q.
 *   Ottenuto con: doca_mmap_get_mkey(rxq_mmap, all_ddevs[q], &mkey)
 *   dopo aver chiamato doca_mmap_add_dev(rxq_mmap, all_ddevs[q]) per ogni q.
 *
 *   La NIC porta q può fare DMA READ dal buffer GPU di questa porta usando
 *   rxq_mkey_for_port[q] come Authorization Key nei propri WQE di trasmissione.
 *   Questo è necessario perché ogni NIC ha un Protection Domain (PD) RDMA
 *   separato: la chiave mkey[self] non è valida per un'altra NIC.
 */
struct bridge_port {
    struct doca_dev *ddev;

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

    /* ── DOCA Flow: BASIC ROOT pipe con match wildcard ───────────────── */
    struct doca_flow_port       *flow_port;
    struct doca_flow_pipe       *root_pipe;
    struct doca_flow_pipe_entry *root_entry;
};

/* =========================================================================
 * STRUTTURA bridge_kernel_params
 * =========================================================================
 * Passata per valore al kernel CUDA tramite il wrapper extern "C".
 * CUDA copia l'intera struct in kernel parameter memory (≤4 KB):
 *   con MAX_N_PORTS=8: ~416 byte, ben entro il limite.
 *
 * rxq_mkey[p]:
 *   mkey del buffer GPU della porta p.
 *   Su BF2 entrambe le porte condividono lo stesso PD RDMA, quindi questo
 *   mkey è valido per qualsiasi TXQ del bridge (zero-copy cross-port).
 *   Calcolato come: ports[p].rxq_mkey
 */
struct bridge_kernel_params {
    int      n_ports;
    struct doca_gpu_eth_rxq *rxq_gpu[MAX_N_PORTS];
    struct doca_gpu_eth_txq *txq_gpu[MAX_N_PORTS];
    uint32_t rxq_mkey_cross[MAX_N_PORTS][MAX_N_PORTS]; /* [src][dst] = mkey rxq[src] per NIC dst */
    uint64_t *mac_table;
    uint32_t *exit_cond;
    uint64_t *fwd_count;

    /* ── Diagnostica live (CPU_GPU, pubblicata una volta per ogni giro
     * completo su tutte le porte, con __threadfence_system) ──────────── */
    uint64_t *rx_pkt_total;   /* [n_ports]: pacchetti ricevuti realmente da RXQ */
    uint64_t *flood_count;    /* pacchetti floodati (dst MAC sconosciuto) */
    uint64_t *unicast_count;  /* pacchetti forwardati per unicast lookup */
    uint64_t *drop_count;     /* pacchetti droppati (dst sulla stessa porta src) */
    uint64_t *wqe_base_dbg;   /* [n_ports]: snapshot di wqe_base[q] (indice WQE
                                * assoluto, sempre crescente — NON si resetta al
                                * wraparound del ring; la posizione fisica è
                                * wqe_base % MAX_SQ_DESCR_NUM). Serve a capire,
                                * se il kernel si blocca dentro poll_completion_at,
                                * a che punto del ring TXQ era rimasto fermo. */
    uint64_t *submit_done_dbg; /* [n_ports]: pubblicato SUBITO DOPO che
                                * doca_gpu_dev_eth_txq_submit() ritorna, PRIMA
                                * di iniziare il poll. Se al freeze wqe_base_dbg
                                * si è aggiornato ma submit_done_dbg no, il
                                * blocco è dentro submit() (es. nello spinlock
                                * doca_gpu_dev_eth_lock); se anche
                                * submit_done_dbg è aggiornato, il blocco è nel
                                * poll dopo, non dentro submit(). */
    uint64_t *batch_round_dbg; /* [n_ports]: numero REALE di batch/round completati
                                * con successo per la TXQ q (incrementato una volta
                                * per esecuzione di Fase 3, esattamente insieme a
                                * cqe_idx[q] += 1). Pubblicato PRIMA del submit/poll
                                * del batch successivo, quindi al momento di un
                                * eventuale freeze riflette l'ultimo round riuscito
                                * (il round bloccato è batch_round_dbg[q]+1). */
    struct mac_flap_record *flap_ring;  /* [FLAP_RING_SIZE] */
    uint32_t *flap_ring_head;           /* contatore monotono totale flap */
};

/* =========================================================================
 * DICHIARAZIONE WRAPPER KERNEL
 * =========================================================================
 * Il wrapper in gpu_bridge_kernel.cu prende il puntatore (convenzione C),
 * lo dereferenzia e passa la struct per valore al kernel CUDA.
 */
#ifdef __cplusplus
extern "C" {
#endif

doca_error_t kernel_launch_bridge(cudaStream_t stream,
                                   struct bridge_kernel_params *kp);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BRIDGE_H */
