/*
 * gpu_bridge_kernel.cu — kernel CUDA per il GPU L2 Bridge a N porte.
 *
 * ARCHITETTURA DEL KERNEL:
 *
 *   Un solo kernel persistente, 1 blocco, 32 thread (= 1 warp NVIDIA).
 *   Il kernel gira per sempre finché la CPU non scrive 1 in exit_cond.
 *
 *   Il loop principale itera su tutte le N porte sorgente:
 *
 *     per ogni porta src (0..n_ports-1):
 *       Riceve da rxq[src], processa, invia verso le porte di uscita.
 *
 *   Per ogni porta sorgente, il processing ha tre fasi:
 *
 *     Fase 1 — inizializzazione contatori batch (solo thread 0):
 *       Reset next_wqe_slot[q] e last_wqe_pkt_*[q] per ogni TXQ.
 *
 *     Fase 2 — parallela (tutti i 32 thread contemporaneamente):
 *       Ogni thread gestisce i propri pacchetti (t, t+32, t+64, ...):
 *         1. Backward learning: src_mac → porta src nella FIB
 *         2. FIB lookup: dst_mac → porta (o -1 per flood)
 *         3. Calcola egress_mask (bitmask porte di uscita)
 *         4. Per ogni bit q in egress_mask:
 *            a. atomicAdd su next_wqe_slot[q] → ottiene slot consecutivo
 *            b. Aggiorna last_wqe_pkt_*[q] (per il fixup NOTIFY)
 *            c. Riempie WQE su txq[q] senza NOTIFY
 *
 *     Fase 3 — seriale (solo thread 0):
 *       Per ogni TXQ q con WQE in questo batch:
 *         a. Ri-riempie l'ULTIMO WQE con flag NOTIFY (fixup)
 *         b. Suona il doorbell (submit)
 *         c. Aspetta il CQE (poll_completion_at)
 *
 * PERCHÉ atomicAdd PER I WQE (insight: la TXQ è un ring circolare):
 *   I WQE devono avere indici consecutivi senza buchi nel ring.
 *   Con atomicAdd ogni thread ottiene uno slot unico e consecutivo
 *   senza dover fare prefix sum o serializzare tutta la fase di fill.
 *   Il ring è circolare: non serve sapere il numero totale di pacchetti
 *   già inviati (al contrario di un buffer lineare dove servirebbe
 *   un offset globale). L'unico vincolo è NOTIFY sul LAST WQE del batch.
 *
 * PROPRIETÀ "LAST WRITE WINS" IN UN SINGOLO WARP:
 *   Con 32 thread = 1 warp (lockstep hardware NVIDIA), le atomicAdd su
 *   next_wqe_slot[q] sono serializzate in lane-order (thread 0, 1, ..., 31).
 *   Il thread con lo slot più alto per TXQ q è anche l'ULTIMO ad aggiornare
 *   last_wqe_pkt_*[q] in quella iterazione del while loop.
 *   Poiché le iterazioni del while loop sono sequenziali (lockstep),
 *   dopo il while loop last_wqe_pkt_*[q] contiene i dati dell'ultimo WQE
 *   (quello con l'indice più alto = quello che thread 0 deve ri-riempire
 *   con NOTIFY). Questa proprietà è valida SOLO per un singolo warp.
 *
 * FLOODING A N PORTE:
 *   egress_mask = ((1u << n_ports) - 1u) & ~(1u << src)
 *   Un pacchetto floodato genera n_ports-1 WQE (uno per ogni altra TXQ).
 *   Ogni WQE usa il mkey di rxq[src] valido per la NIC della porta di uscita.
 *   Con 8 porte e 2048 pacchetti floodati: 14336 WQE totali.
 *
 * ZERO-COPY CROSS-PORT:
 *   I WQE di txq[q] puntano direttamente ai dati in GPU memory di rxq[src].
 *   La NIC porta q fa DMA dalla GPU senza nessuna copia CPU/GPU.
 *   Il mkey rxq_mkey_cross[src][q] autorizza la NIC q ad accedere
 *   al buffer GPU della porta src (PD RDMA separato per ogni NIC).
 */

#include <stdio.h>

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gpu_bridge.h"

/* =========================================================================
 * FUNZIONI DEVICE — MAC TABLE (FIB)
 * =========================================================================
 */

/*
 * mac_to_u64: converte 6 byte MAC (network order) in uint64_t (bit 0-47).
 * Byte [0] = più significativo. Bit 48-63 sempre zero.
 */
static __device__ __forceinline__ uint64_t mac_to_u64(const uint8_t *mac)
{
    return ((uint64_t)mac[0] << 40) |
           ((uint64_t)mac[1] << 32) |
           ((uint64_t)mac[2] << 24) |
           ((uint64_t)mac[3] << 16) |
           ((uint64_t)mac[4] <<  8) |
            (uint64_t)mac[5];
}

/*
 * mac_hash: hash FNV-1a a 32 bit sul MAC a 48 bit.
 * Veloce su GPU (no divisione), buona distribuzione su input corti.
 * Risultato mascherato con (MAC_TABLE_SIZE - 1) = 4095 (AND, non modulo).
 */
static __device__ __forceinline__ uint32_t mac_hash(uint64_t mac48)
{
    uint32_t h = 2166136261u;
    h ^= (uint8_t)(mac48 >> 40); h *= 16777619u;
    h ^= (uint8_t)(mac48 >> 32); h *= 16777619u;
    h ^= (uint8_t)(mac48 >> 24); h *= 16777619u;
    h ^= (uint8_t)(mac48 >> 16); h *= 16777619u;
    h ^= (uint8_t)(mac48 >>  8); h *= 16777619u;
    h ^= (uint8_t)(mac48);       h *= 16777619u;
    return h & (MAC_TABLE_SIZE - 1);
}

/*
 * mac_learn: registra nella FIB che mac48 è raggiungibile sulla porta src_port.
 *
 * FORMATO ENTRY (uint64_t):
 *   bit 63:    valid (1 = slot occupato)
 *   bit 48-55: porta (0..MAX_N_PORTS-1)
 *   bit 0-47:  MAC address
 *
 * ALGORITMO — linear probing con atomici (thread-safe):
 *   slot vuoto:      atomicCAS(0 → nuova_entry) per inserire
 *   stesso MAC:      atomicExch per aggiornare la porta
 *   MAC diverso:     linear probe al prossimo slot
 *   tabella piena:   rinuncia silenziosamente (flood come fallback)
 */
static __device__ void mac_learn(uint64_t *mac_table, uint64_t mac48, int src_port)
{
    uint64_t nuova_entry = MAC_ENTRY_VALID_BIT |
                           ((uint64_t)(src_port & 0xFF) << MAC_ENTRY_PORT_SHIFT) |
                           (mac48 & MAC_48BIT_MASK);
    uint32_t slot = mac_hash(mac48);

    for (int probe = 0; probe < MAC_TABLE_PROBE_MAX; probe++) {
        uint64_t val = *(volatile uint64_t *)&mac_table[slot];

        if (val == 0) {
            uint64_t old = atomicCAS((unsigned long long *)&mac_table[slot],
                                     0ULL,
                                     (unsigned long long)nuova_entry);
            if (old == 0)
                return;
            val = old;
        }

        if ((val & MAC_48BIT_MASK) == (mac48 & MAC_48BIT_MASK)) {
            atomicExch((unsigned long long *)&mac_table[slot],
                       (unsigned long long)nuova_entry);
            return;
        }

        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
}

/*
 * mac_lookup: cerca mac48 nella FIB.
 * Ritorna la porta (0..n_ports-1) o -1 (MAC sconosciuto = flood).
 * Lettura volatile: vede le scritture atomiche degli altri thread.
 */
static __device__ int mac_lookup(const uint64_t *mac_table, uint64_t mac48)
{
    uint32_t slot = mac_hash(mac48);

    for (int probe = 0; probe < MAC_TABLE_PROBE_MAX; probe++) {
        uint64_t entry = *(volatile uint64_t *)&mac_table[slot];

        if (entry == 0)
            return -1;

        if ((entry & MAC_ENTRY_VALID_BIT) &&
            (entry & MAC_48BIT_MASK) == (mac48 & MAC_48BIT_MASK)) {
            /* Estrae 8 bit per la porta (supporta fino a 256 porte) */
            return (int)((entry & MAC_ENTRY_PORT_MASK) >> MAC_ENTRY_PORT_SHIFT);
        }

        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
    return -1;
}

/* =========================================================================
 * KERNEL PRINCIPALE — bridge_kernel
 * =========================================================================
 *
 * Dimensioni: <<<1, 32, 0, stream>>> (1 blocco, 32 thread = 1 warp)
 *
 * Il kernel riceve struct bridge_kernel_params PER VALORE.
 * CUDA copia l'intera struct nella kernel parameter memory (≤4 KB).
 * I puntatori interni (rxq_gpu, txq_gpu, mac_table, ...) puntano a
 * memoria GPU o handle validi nel device, quindi sono validi nel kernel.
 *
 * SHARED MEMORY (condivisa da tutti i 32 thread):
 *   rx_first_pkt_idx, rx_pkt_count, rx_attr[]:
 *     Riempiti da doca_gpu_dev_eth_rxq_recv<BLOCK> cooperativamente.
 *     rx_attr[i].bytes = lunghezza del pacchetto i.
 *
 *   next_wqe_slot[MAX_N_PORTS]:
 *     Contatore atomico per slot WQE per ogni TXQ.
 *     Ogni thread fa atomicAdd per ottenere il proprio slot consecutivo.
 *     Azzerato all'inizio di ogni batch (per ogni porta src).
 *
 *   last_wqe_pkt_addr/bytes/mkey[MAX_N_PORTS]:
 *     Info dell'ultimo WQE scritto per ogni TXQ.
 *     Thread 0 le usa dopo __syncthreads() per ri-riempire l'ultimo
 *     WQE con flag NOTIFY (fixup prima del submit).
 *     "Last write wins" è garantito nel singolo warp (vedi header).
 *
 * VARIABILI PRIVATE DI THREAD 0 (solo usate in Fase 3):
 *   wqe_base[MAX_N_PORTS]: prossimo indice WQE per ogni TXQ (persistente tra batch)
 *   cqe_idx[MAX_N_PORTS]:  prossimo indice CQE per ogni TXQ (persistente tra batch)
 *   tot_fwd:               totale pacchetti inoltrati
 */
__global__ void bridge_kernel(struct bridge_kernel_params kp)
{
    /* ── Shared memory ─────────────────────────────────────────────────── */
    __shared__ uint64_t rx_first_pkt_idx;
    __shared__ uint32_t rx_pkt_count;
    __shared__ struct doca_gpu_dev_eth_rxq_attr rx_attr[MAX_RX_NUM_PKTS];

    /*
     * next_wqe_slot[q]: contatore atomico per slot WQE su txq[q].
     * Ogni thread fa atomicAdd per reclamare il proprio slot nel ring.
     * Reset a zero prima di ogni batch.
     */
    __shared__ uint32_t next_wqe_slot[MAX_N_PORTS];

    /*
     * Info dell'ultimo WQE per ogni TXQ (vedi commento nel file header).
     * Aggiornate con semplici scritture (no atomic): la proprietà di lockstep
     * del singolo warp garantisce che l'ultima scrittura corrisponda all'ultimo
     * WQE (quello con l'indice più alto nel ring).
     */
    __shared__ uint64_t last_wqe_pkt_addr[MAX_N_PORTS];
    __shared__ uint32_t last_wqe_pkt_bytes[MAX_N_PORTS];
    __shared__ uint32_t last_wqe_pkt_mkey[MAX_N_PORTS];

    /*
     * wqe_base[q]: indice assoluto del prossimo WQE da scrivere per TXQ q.
     * Deve essere __shared__ perché tutti i thread lo leggono in Fase 2B
     * per calcolare l'indice assoluto del proprio WQE (wqe_base[q] + slot).
     * Thread 0 lo aggiorna in Fase 3 dopo ogni batch.
     * Se fosse locale, thread 1-31 avrebbero sempre wqe_base=0 e
     * dal secondo batch in poi scriverebbero i WQE all'indice sbagliato.
     */
    __shared__ uint64_t wqe_base[MAX_N_PORTS];

    /* cqe_idx e tot_fwd: acceduti solo da thread 0, possono stare in locale */
    uint64_t cqe_idx[MAX_N_PORTS];
    uint64_t tot_fwd = 0;
    doca_error_t ret;

    if (threadIdx.x == 0) {
        for (int q = 0; q < MAX_N_PORTS; q++) {
            wqe_base[q] = 0;
            cqe_idx[q]  = 0;
        }
    }
    __syncthreads();

    /* =====================================================================
     * LOOP PRINCIPALE
     * ===================================================================== */
    while (DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) == 0) {

        /* Loop su tutte le porte sorgente (0, 1, ..., n_ports-1) */
        for (int src = 0;
             src < kp.n_ports && DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) == 0;
             src++)
        {
            /* ==============================================================
             * FASE 1: RICEZIONE
             * Tutti i 32 thread cooperano (EXEC_SCOPE_BLOCK, 1 warp intero).
             * Aspetta ≥1 pacchetto o MAX_RX_TIMEOUT_NS (500 µs).
             * ============================================================== */
            ret = doca_gpu_dev_eth_rxq_recv<
                    DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
                    DOCA_GPUNETIO_ETH_MCST_AUTO,
                    DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
                    DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
                kp.rxq_gpu[src],
                MAX_RX_NUM_PKTS,
                MAX_RX_TIMEOUT_NS,
                &rx_first_pkt_idx,
                &rx_pkt_count,
                rx_attr);

            if (ret != DOCA_SUCCESS) {
                if (threadIdx.x == 0)
                    DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) = 1;
                break;
            }

            if (rx_pkt_count == 0)
                continue;

            /* ==============================================================
             * FASE 2A: RESET CONTATORI BATCH (solo thread 0)
             * ============================================================== */
            if (threadIdx.x == 0) {
                for (int q = 0; q < kp.n_ports; q++) {
                    next_wqe_slot[q]      = 0;
                    last_wqe_pkt_addr[q]  = 0;
                    last_wqe_pkt_bytes[q] = 0;
                    last_wqe_pkt_mkey[q]  = 0;
                }
            }
            __syncthreads();

            /* ==============================================================
             * FASE 2B: BACKWARD LEARNING + FIB LOOKUP + WQE FILL PARALLELO
             *
             * Tutti i 32 thread in parallelo.
             * Thread t gestisce pacchetti t, t+32, t+64, ...
             *
             * Per ogni pacchetto:
             *   1. mac_learn(src_mac, src): impara la sorgente
             *   2. mac_lookup(dst_mac): trova destinazione
             *   3. Calcola egress_mask:
             *      - drop (dst_port==src): mask=0
             *      - flood (dst_port==-1): tutti i bit tranne src
             *      - unicast:              solo bit dst_port
             *   4. Per ogni bit q in egress_mask:
             *      - atomicAdd → slot consecutivo su txq[q]
             *      - aggiorna last_wqe_pkt_*[q] (last write wins in warp)
             *      - riempi WQE senza NOTIFY
             * ============================================================== */
            uint32_t pkt_idx = threadIdx.x;
            while (pkt_idx < rx_pkt_count) {

                uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                    kp.rxq_gpu[src], rx_first_pkt_idx + pkt_idx);
                const uint8_t *eth = (const uint8_t *)(uintptr_t)pkt_addr;

                uint64_t dst_mac48 = mac_to_u64(eth);       /* byte 0-5 */
                uint64_t src_mac48 = mac_to_u64(eth + 6);   /* byte 6-11 */

                /* Backward learning: il mittente è raggiungibile su 'src' */
                mac_learn(kp.mac_table, src_mac48, src);

                /* FIB lookup: verso quale porta mandare dst_mac? */
                int dst_port = mac_lookup(kp.mac_table, dst_mac48);

                /*
                 * Calcolo egress_mask (uint32_t, bit q = "invia a porta q"):
                 *
                 *   dst_port == src  → drop: il destinatario è sulla porta
                 *                      di ingresso (loop potenziale)
                 *   dst_port == -1   → flood: manda a tutte le porte tranne src
                 *                      ((1<<n_ports)-1) & ~(1<<src)
                 *   altrimenti       → unicast: solo porta dst_port
                 *
                 * Con 2 porte: flood e unicast verso l'altra porta coincidono.
                 * Con N>2 porte: flood invia a N-1 TXQ, unicast a 1 sola.
                 */
                uint32_t egress_mask;
                if (dst_port == src) {
                    egress_mask = 0u;
                } else if (dst_port < 0) {
                    egress_mask = ((1u << kp.n_ports) - 1u) & ~(1u << src);
                } else {
                    egress_mask = (1u << dst_port);
                }

                /*
                 * Per ogni porta di uscita q (bit settato in egress_mask):
                 *
                 * atomicAdd su next_wqe_slot[q]:
                 *   Nel singolo warp (lockstep), l'atomicAdd è serializzata
                 *   in lane-order. Ogni thread ottiene uno slot unico e
                 *   consecutivo nel ring della TXQ q.
                 *   Non serve prefix sum: il ring è circolare, non c'è
                 *   bisogno di sapere quanti WQE ci sono stati prima.
                 *
                 * last_wqe_pkt_*[q]:
                 *   Scrittura semplice (no atomic). Nel lockstep del warp,
                 *   l'ultimo thread a scrivere per TXQ q ha anche lo slot
                 *   più alto (perché l'atomicAdd assegna slot in lane-order
                 *   e la scrittura segue immediatamente l'atomicAdd nella
                 *   stessa "istruzione lockstep"). Vedere commento nel header.
                 *
                 * WQE zero-copy cross-port:
                 *   Il WQE punta a pkt_addr nel buffer GPU della porta src.
                 *   La NIC porta q legge direttamente dalla VRAM tramite PCIe.
                 *   Nessuna copia: zero overhead di memoria.
                 */
                for (int q = 0; q < kp.n_ports; q++) {
                    if (!(egress_mask & (1u << q)))
                        continue;

                    uint32_t mkey = kp.rxq_mkey_cross[src][q];
                    uint32_t slot = atomicAdd(&next_wqe_slot[q], 1u);

                    /* Last write wins: il thread con slot più alto scrive ultimo */
                    last_wqe_pkt_addr[q]  = pkt_addr;
                    last_wqe_pkt_bytes[q] = rx_attr[pkt_idx].bytes;
                    last_wqe_pkt_mkey[q]  = mkey;

                    struct doca_gpu_dev_eth_txq_wqe *wqe =
                        doca_gpu_dev_eth_txq_get_wqe_ptr(
                            kp.txq_gpu[q], wqe_base[q] + slot);
                    doca_gpu_dev_eth_txq_wqe_prepare_send(
                        kp.txq_gpu[q], wqe, wqe_base[q] + slot,
                        pkt_addr, mkey, rx_attr[pkt_idx].bytes,
                        DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);
                }

                pkt_idx += blockDim.x;
            }

            /* Tutti i WQE sono stati scritti. */
            __syncthreads();

            /* ==============================================================
             * FASE 3: FIXUP NOTIFY + SUBMIT + POLL (solo thread 0)
             *
             * Per ogni TXQ q con WQE in questo batch:
             *   a. next_wqe_slot[q] = n = numero totale di WQE per txq[q]
             *   b. Ri-riempie l'ULTIMO WQE (indice wqe_base[q]+n-1) con NOTIFY.
             *      La NIC genera una CQE SOLO per WQE con NOTIFY: invece di
             *      una CQE per WQE (enorme overhead), otteniamo una CQE per batch.
             *   c. doca_gpu_dev_eth_txq_submit: suona il doorbell → NIC inizia DMA
             *   d. doca_gpu_dev_eth_txq_poll_completion_at: aspetta CQE
             *   e. Aggiorna wqe_base[q] e cqe_idx[q] per il prossimo batch
             *
             * Con flooding a N porte, thread 0 fa N-1 submit+poll consecutivi.
             * (Ottimizzazione futura: sottomettere tutti prima di pollare,
             *  ma richiederebbe tracking separato dei doorbell. Non necessario ora.)
             * ============================================================== */
            if (threadIdx.x == 0) {
                for (int q = 0; q < kp.n_ports; q++) {
                    uint32_t n = next_wqe_slot[q];
                    if (n == 0)
                        continue;

                    /*
                     * Fixup: ri-riempie l'ultimo WQE (slot più alto) con NOTIFY.
                     * last_wqe_pkt_*[q] contiene i dati del WQE con slot n-1
                     * (garantito dalla proprietà "last write wins" nel warp).
                     */
                    uint64_t last_slot = wqe_base[q] + n - 1;
                    struct doca_gpu_dev_eth_txq_wqe *last_wqe =
                        doca_gpu_dev_eth_txq_get_wqe_ptr(kp.txq_gpu[q], last_slot);
                    doca_gpu_dev_eth_txq_wqe_prepare_send(
                        kp.txq_gpu[q], last_wqe, last_slot,
                        last_wqe_pkt_addr[q],
                        last_wqe_pkt_mkey[q],
                        last_wqe_pkt_bytes[q],
                        DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY);

                    /* Doorbell: dice alla NIC q "ci sono n WQE da processare" */
                    doca_gpu_dev_eth_txq_submit(kp.txq_gpu[q], wqe_base[q] + n);

                    /* Aspetta la CQE generata dal WQE con NOTIFY */
                    ret = doca_gpu_dev_eth_txq_poll_completion_at<
                            DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
                        kp.txq_gpu[q], cqe_idx[q],
                        DOCA_GPUNETIO_ETH_WAIT_FLAG_B);

                    if (ret != DOCA_SUCCESS) {
                        DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) = 1;
                        break;
                    }

                    wqe_base[q] += n;
                    cqe_idx[q]  += 1;
                    tot_fwd     += n;
                }
            }
            __syncthreads();

        } /* for src */

    } /* while (!exit_cond) */

    /*
     * Pubblica il contatore verso la CPU.
     * __threadfence_system(): flush cache GPU attraverso PCIe.
     * Necessario perché fwd_count è in CPU_GPU memory (pinned sul lato CPU);
     * senza questo la CPU potrebbe leggere un valore stale dalla propria cache.
     */
    if (threadIdx.x == 0) {
        *kp.fwd_count = tot_fwd;
        __threadfence_system();
    }
}

/* =========================================================================
 * WRAPPER EXTERN "C" — chiamabile da gpu_bridge.c (codice C)
 * =========================================================================
 * Prende kp per puntatore (convenzione C), lo dereferenzia e passa la struct
 * per valore al kernel CUDA. CUDA copia automaticamente i ~400 B in
 * kernel parameter memory.
 */
extern "C" {

doca_error_t kernel_launch_bridge(cudaStream_t stream,
                                   struct bridge_kernel_params *kp)
{
    cudaError_t cuda_ret;

    if (!kp || !kp->mac_table || !kp->exit_cond || !kp->fwd_count)
        return DOCA_ERROR_INVALID_VALUE;

    if (kp->n_ports < 2 || kp->n_ports > MAX_N_PORTS)
        return DOCA_ERROR_INVALID_VALUE;

    for (int i = 0; i < kp->n_ports; i++) {
        if (!kp->rxq_gpu[i] || !kp->txq_gpu[i])
            return DOCA_ERROR_INVALID_VALUE;
    }

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA error before kernel launch: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    /* Passa la struct per valore: CUDA copia i ~400 B in parameter memory */
    bridge_kernel<<<1, BRIDGE_BLOCK_THREADS, 0, stream>>>(*kp);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern "C" */
