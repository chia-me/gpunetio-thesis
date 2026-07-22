/*
 * gpu_bridge.h — strutture, costanti e dichiarazioni per il GPU L2 Bridge.
 *
 * Il bridge collega due porte fisiche della BF2 (ad:00.0 e ad:00.1).
 * Ogni pacchetto che arriva su una porta viene processato dalla A30X:
 *   - backward learning: il src_mac viene associato alla porta di ingresso
 *   - FIB lookup: si cerca la porta di uscita per il dst_mac
 *   - forward/drop: il pacchetto viene inviato sull'altra porta o droppato
 * Tutto avviene nella GPU senza coinvolgimento del CPU per singolo pacchetto.
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
 * COSTANTI — CODA DI RICEZIONE (RXQ)
 * =========================================================================
 *
 * MAX_PKT_NUM: slot nel ring buffer ciclico della RXQ.
 *   La BF2 DMA i pacchetti in questo buffer circolare in GPU memory.
 *   16384 slot × MAX_PKT_SIZE byte = 32 MB per porta.
 *
 * MAX_PKT_SIZE: dimensione massima per pacchetto, in byte.
 *   2048 è sufficiente per Ethernet standard (MTU 1500 + header 14B + padding).
 *
 * MAX_RX_TIMEOUT_NS: quanto aspetta la recv del kernel prima di tornare a mani
 *   vuote. 500 µs è un buon compromesso: risposta rapida al SIGINT ma non così
 *   breve da sprecare cicli GPU in loop vuoti.
 *
 * MAX_RX_NUM_PKTS: quanti pacchetti può restituire una singola recv call.
 *   Con EXEC_SCOPE_BLOCK e 32 thread, ogni thread processa fino a
 *   MAX_RX_NUM_PKTS / 32 = 64 pacchetti in parallelo.
 */
#define MAX_PKT_NUM       16384
#define MAX_PKT_SIZE      2048
#define MAX_RX_TIMEOUT_NS 500000      /* 500 microsecondi */
#define MAX_RX_NUM_PKTS   2048

/* =========================================================================
 * COSTANTI — CODA DI TRASMISSIONE (TXQ)
 * =========================================================================
 *
 * MAX_SQ_DESCR_NUM: numero di Work Queue Entry (WQE) nel Send Queue della TXQ.
 *   Ogni WQE = istruzione per la NIC "invia questi N byte da questo indirizzo".
 *   1024 WQE è abbondante per i burst tipici (max MAX_RX_NUM_PKTS = 2048,
 *   ma i WQE si consumano in ordine e la NIC li processa velocemente).
 */
#define MAX_SQ_DESCR_NUM  1024

/* =========================================================================
 * COSTANTI — DOCA FLOW
 * =========================================================================
 *
 * FLOW_NB_COUNTERS: numero di contatori hardware per statistiche sulle pipe.
 */
#define FLOW_NB_COUNTERS  512

/* =========================================================================
 * COSTANTI — KERNEL CUDA
 * =========================================================================
 *
 * BRIDGE_BLOCK_THREADS: thread nel blocco CUDA del kernel bridge.
 *   Deve essere esattamente 32 = 1 warp NVIDIA.
 *
 *   Perché 32? EXEC_SCOPE_BLOCK con 32 thread = 1 warp intero.
 *   La chiamata doca_gpu_dev_eth_rxq_recv<BLOCK> usa TUTTI i 32 thread
 *   cooperativamente per ricevere il batch. Con esattamente 1 warp,
 *   la sincronizzazione è garantita dall'hardware (warp execution) senza
 *   overhead di __syncthreads sul percorso critico della recv.
 */
#define BRIDGE_BLOCK_THREADS 32

/* =========================================================================
 * COSTANTI — MAC TABLE (FIB)
 * =========================================================================
 *
 * MAC_TABLE_SIZE: numero di slot nella hash table.
 *   Deve essere potenza di 2: usiamo (h & (SIZE-1)) invece di (h % SIZE),
 *   che è molto più veloce sulla GPU (AND vs divisione).
 *   4096 = 4K entry: sufficiente per una LAN tipica.
 *
 * MAC_TABLE_PROBE_MAX: max probe steps nel linear probing prima di rinunciare.
 *   256 significa che esploriamo fino a 256 slot consecutivi prima di dichiarare
 *   la tabella "piena" per quel bucket. Nella pratica con 4K slot e buona
 *   distribuzione hash, la collision chain è raramente > 4-5.
 *
 * MAC_ENTRY_VALID_BIT: bit 63 di un'entry uint64_t = slot occupato.
 *   Questo permette di distinguere slot vuoto (entry==0) da un entry per
 *   il MAC 00:00:00:00:00:00 (entry = VALID_BIT | 0 = solo bit 63 acceso).
 *
 * MAC_ENTRY_PORT_SHIFT: il numero di porta (0 o 1) è al bit 48.
 *   Bit 63:    valid flag
 *   Bit 48:    porta (0 = prima porta BF2, 1 = seconda porta BF2)
 *   Bit 0-47:  MAC address a 48 bit
 */
#define MAC_TABLE_SIZE       4096
#define MAC_TABLE_PROBE_MAX  256
#define MAC_ENTRY_VALID_BIT  ((uint64_t)1 << 63)
#define MAC_ENTRY_PORT_SHIFT 48
#define MAC_48BIT_MASK       0x0000FFFFFFFFFFFFULL

/* =========================================================================
 * COSTANTI — NUMERO DI PORTE
 * =========================================================================
 */
#define BRIDGE_NUM_PORTS 2

/* =========================================================================
 * MACRO — ALLINEAMENTO MEMORIA
 * =========================================================================
 * doca_gpu_mem_alloc richiede che la dimensione sia multiplo della page size.
 */
#define ALIGN_UP(sz, align) (((sz) + (align) - 1) / (align) * (align))

/* =========================================================================
 * STRUTTURA bridge_port
 * =========================================================================
 * Rappresenta UNA delle due porte fisiche del bridge.
 * Il main alloca un array: struct bridge_port ports[BRIDGE_NUM_PORTS].
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  FLUSSO DEI DATI PER UNA PORTA (es. porta 0)                        │
 * │                                                                      │
 * │  [Rete] ──DMA_write──> [rxq_buf in GPU memory A30X]                  │
 * │                               │                                      │
 * │                        kernel CUDA legge                             │
 * │                               │                                      │
 * │                        [WQE su TXQ1] ──DMA_read──> [Rete porta 1]   │
 * │                                                                      │
 * │  I WQE di TXQ1 puntano a rxq_buf di porta 0 con rxq_mkey_for_other  │
 * │  (cross-port zero-copy: nessuna copia dei dati)                      │
 * └─────────────────────────────────────────────────────────────────────┘
 */
struct bridge_port {

    /* ── Device DOCA della NIC per questa porta ─────────────────────────
     * Aperto con doca_dev_open() tramite PCI address (es. "ad:00.0").
     * Usato per creare RXQ, TXQ e la porta DOCA Flow.
     */
    struct doca_dev *ddev;

    /* ── Coda di Ricezione (RXQ) ──────────────────────────────────────
     *
     * La NIC BF2 riceve pacchetti dalla rete e li scrive DIRETTAMENTE
     * nella GPU memory (rxq_buf) tramite DMA peer-to-peer (gdrcopy/dmabuf).
     * Nessun coinvolgimento della CPU: la NIC parla direttamente con la A30X.
     *
     * rxq_cpu:  handle usato durante il setup (da questo file .c)
     * rxq_gpu:  handle passato al kernel CUDA (puntatore a struttura in GPU)
     * rxq_mmap: descrive rxq_buf alla NIC tramite RDMA registration
     * rxq_buf:  il buffer ciclico in GPU memory dove arrivano i pacchetti
     * rxq_dmabuf_fd: file descriptor per DMABuf (se supportato dal kernel)
     *
     * rxq_mkey_for_own_txq:   mkey per la NIC di QUESTA porta.
     *   Usato quando la TXQ di questa porta legge dal proprio rxq_buf
     *   (non usato nel bridge normale, ma disponibile per loopback/debug).
     *
     * rxq_mkey_for_other_txq: mkey per la NIC dell'ALTRA porta.
     *   Questo è il mkey CRUCIALE per il forwarding cross-port zero-copy:
     *   la TXQ dell'altra porta usa questo mkey per leggere da rxq_buf
     *   di questa porta.
     *
     *   Perché servono due mkey diversi?
     *   In InfiniBand/RDMA, ogni dispositivo NIC ha un proprio Protection
     *   Domain (PD) con chiavi di accesso separate. La porta 1 non può usare
     *   il mkey della porta 0 e viceversa. Per permettere alla porta 1 di
     *   leggere il buffer GPU della porta 0, registriamo quel buffer in
     *   entrambi i PD: doca_mmap_add_dev(mmap, ddev0) + doca_mmap_add_dev(mmap, ddev1),
     *   poi doca_mmap_get_mkey(mmap, ddev1) ci dà il mkey corretto per la porta 1.
     */
    struct doca_ctx         *rxq_ctx;
    struct doca_eth_rxq     *rxq_cpu;
    struct doca_gpu_eth_rxq *rxq_gpu;
    struct doca_mmap        *rxq_mmap;
    void                    *rxq_buf;
    int                      rxq_dmabuf_fd;
    size_t                   rxq_buf_size;      /* dimensione allocata (per cleanup) */
    uint32_t                 rxq_mkey_for_own_txq;    /* mkey per questa NIC */
    uint32_t                 rxq_mkey_for_other_txq;  /* mkey per l'altra NIC */

    /* ── Coda di Trasmissione (TXQ) ───────────────────────────────────
     *
     * Il kernel CUDA scrive WQE (Work Queue Entry) nella TXQ.
     * Ogni WQE contiene: [indirizzo GPU sorgente] [mkey] [dimensione] [flags].
     * La NIC legge i WQE e fa DMA dalla GPU verso la rete.
     *
     * IMPORTANTE: la TXQ NON ha un proprio buffer dati.
     * I WQE puntano al buffer rxq_buf dell'ALTRA porta (zero-copy cross-port).
     *
     * txq_cpu:  handle usato nel setup
     * txq_gpu:  handle passato al kernel CUDA
     */
    struct doca_ctx         *txq_ctx;
    struct doca_eth_txq     *txq_cpu;
    struct doca_gpu_eth_txq *txq_gpu;

    /* ── DOCA Flow ─────────────────────────────────────────────────────
     *
     * DOCA Flow è il sistema di hardware packet steering della BF2.
     * Instrada i pacchetti in arrivo verso la GPU RXQ.
     *
     * Per il bridge vogliamo catturare TUTTO il traffico L2:
     *   - IPv4  (ethertype 0x0800)
     *   - ARP   (ethertype 0x0806)
     *   - IPv6  (ethertype 0x86DD)
     *   - qualsiasi altro ethertype
     *
     * Usiamo una singola BASIC ROOT pipe con match template = {0} (tutti
     * i campi a zero = wildcard su qualsiasi campo) e una sola entry che
     * cattura qualsiasi pacchetto e lo manda alla RSS queue 0 (la GPU RXQ).
     *
     * flow_port:  handle DOCA Flow per questa porta (ID univoco 0 o 1)
     * root_pipe:  la BASIC ROOT pipe con match wildcard
     * root_entry: l'unica entry wildcard nella root pipe
     */
    struct doca_flow_port       *flow_port;
    struct doca_flow_pipe       *root_pipe;
    struct doca_flow_pipe_entry *root_entry;
};

/* =========================================================================
 * STRUTTURA bridge_kernel_params
 * =========================================================================
 * Raggruppa tutti i parametri da passare al kernel CUDA.
 * Allocata sullo stack del main (memoria CPU), i puntatori interni puntano
 * a memoria GPU o handle GPU-side.
 */
struct bridge_kernel_params {
    struct doca_gpu_eth_rxq *rxq_gpu[BRIDGE_NUM_PORTS]; /* handle GPU delle RXQ */
    struct doca_gpu_eth_txq *txq_gpu[BRIDGE_NUM_PORTS]; /* handle GPU delle TXQ */

    /* mkey_for_other[i]: mkey del buffer rxq della porta i,
     * valido per la NIC dell'altra porta (per zero-copy cross-port TX).
     *   mkey_for_other[0] = rxq_mkey_for_other_txq di ports[0]
     *                      → usato dai WQE di txq1 per leggere il buffer di rxq0
     *   mkey_for_other[1] = rxq_mkey_for_other_txq di ports[1]
     *                      → usato dai WQE di txq0 per leggere il buffer di rxq1
     */
    uint32_t rxq_mkey_for_other[BRIDGE_NUM_PORTS];

    uint64_t *mac_table;   /* hash table FIB in GPU memory (cudaMalloc) */
    uint32_t *exit_cond;   /* 0=running, 1=stop (GPU_CPU: GPU la legge veloce, CPU la scrive) */
    uint64_t *fwd_count;   /* contatore forward (CPU_GPU: CPU la legge veloce dopo sync) */
};

/* =========================================================================
 * DICHIARAZIONE DEL WRAPPER KERNEL (definito in gpu_bridge_kernel.cu)
 * =========================================================================
 * extern "C": necessario perché il .cu usa C++ name mangling ma il .c no.
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
