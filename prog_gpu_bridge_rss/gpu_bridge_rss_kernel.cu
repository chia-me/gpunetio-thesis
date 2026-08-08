/*
 * gpu_bridge_rss_kernel.cu — kernel CUDA per il GPU L2 Bridge a N porte
 * con RSS multi-coda (v2).
 *
 * ARCHITETTURA DEL KERNEL (differenza chiave rispetto a v1):
 *
 *   v1: 1 blocco persistente, 32 thread, che itera in un loop seriale su
 *       TUTTE le porte sorgente (for src in 0..n_ports).
 *
 *   v2: griglia di (n_ports * N_QUEUES_PER_PORT) blocchi persistenti,
 *       32 thread ciascuno. Ogni blocco è fissato UNA VOLTA per tutte a
 *       una coppia (porta sorgente, coda RSS) tramite blockIdx.x, letta
 *       da kp.queues[blockIdx.x]. Il blocco riceve SOLO dalla propria
 *       rxq e trasmette SOLO sulle proprie TXQ private (una per porta di
 *       destinazione) — quindi il loop "for src" di v1 sparisce: non
 *       serve più, ogni blocco È già dedicato a una sola sorgente.
 *
 *   Per ogni blocco, il processing di un batch ricevuto ha le stesse tre
 *   fasi di v1 (invariate concettualmente, vedi commenti lì per i
 *   dettagli sugli atomici):
 *
 *     Fase 1 — ricezione cooperativa (32 thread, EXEC_SCOPE_BLOCK)
 *     Fase 2 — parallela: backward learning + FIB lookup + WQE fill
 *              (atomicAdd su next_wqe_slot[d], atomicMax su last_wqe_key[d])
 *     Fase 3 — seriale (thread 0): fixup NOTIFY + submit + poll per ogni
 *              TXQ d con WQE in questo batch
 *
 * PERCHÉ NESSUNA SINCRONIZZAZIONE CROSS-BLOCK È NECESSARIA:
 *   In v1 un solo blocco possiede TUTTE le TXQ (una per porta), quindi gli
 *   atomici su next_wqe_slot/last_wqe_key bastano perché tutti i thread che
 *   scrivono sulla stessa TXQ sono nello stesso blocco (shared memory).
 *   In v2, con più blocchi, la stessa garanzia si ottiene NON condividendo
 *   MAI una TXQ fisica tra blocchi diversi: ogni blocco (porta, coda) ha un
 *   set privato di TXQ verso ogni possibile porta di destinazione (vedi
 *   setup lato CPU in gpu_bridge_rss.c). Risultato: gli stessi identici
 *   pattern di atomici di v1 restano corretti, ma ora sono __shared__ per
 *   blocco (ogni blocco ha la propria copia) invece che globali per porta.
 *
 * COSA RESTA GLOBALE (condiviso tra TUTTI i blocchi):
 *   - mac_table: la FIB è UNA SOLA, condivisa da tutte le porte/code, con
 *     accesso già thread-safe via atomicCAS/atomicExch/atomicMax (v1).
 *     Funziona identico con più blocchi: gli atomici CUDA sono validi
 *     grid-wide, non solo block-wide.
 *   - flap_ring/flap_ring_head, fwd_count, flood/unicast/drop_count,
 *     rx_pkt_total: contatori diagnostici aggregati con atomicAdd da
 *     thread 0 di ogni blocco, una volta per batch (non per pacchetto),
 *     quindi l'overhead di contesa tra blocchi è trascurabile.
 */

#include <stdio.h>

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gpu_bridge_rss.h"

/* =========================================================================
 * FUNZIONI DEVICE — MAC TABLE (FIB), invariate rispetto a v1
 * ========================================================================= */

static __device__ __forceinline__ uint64_t mac_to_u64(const uint8_t *mac)
{
    return ((uint64_t)mac[0] << 40) |
           ((uint64_t)mac[1] << 32) |
           ((uint64_t)mac[2] << 24) |
           ((uint64_t)mac[3] << 16) |
           ((uint64_t)mac[4] <<  8) |
            (uint64_t)mac[5];
}

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

static __device__ void mac_learn(uint64_t *mac_table, uint64_t mac48, int src_port,
                                  struct mac_flap_record *flap_ring,
                                  uint32_t *flap_ring_head)
{
    if (mac48 & ((uint64_t)1 << 40))
        return;

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
            int old_port = (int)((val & MAC_ENTRY_PORT_MASK) >> MAC_ENTRY_PORT_SHIFT);

            if (old_port != src_port) {
                uint32_t idx = atomicAdd(flap_ring_head, 1u);
                struct mac_flap_record *rec = &flap_ring[idx % FLAP_RING_SIZE];
                rec->mac48    = mac48;
                rec->old_port = (uint32_t)old_port;
                rec->new_port = (uint32_t)src_port;
                rec->seq      = idx;
            }

            atomicExch((unsigned long long *)&mac_table[slot],
                       (unsigned long long)nuova_entry);
            return;
        }

        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
}

static __device__ __forceinline__ bool is_8021d_reserved(uint64_t mac48)
{
    return (mac48 & 0xFFFFFFFFFFF0ULL) == 0x0180C2000000ULL;
}

static __device__ int mac_lookup(const uint64_t *mac_table, uint64_t mac48)
{
    uint32_t slot = mac_hash(mac48);

    for (int probe = 0; probe < MAC_TABLE_PROBE_MAX; probe++) {
        uint64_t entry = *(volatile uint64_t *)&mac_table[slot];

        if (entry == 0)
            return -1;

        if ((entry & MAC_ENTRY_VALID_BIT) &&
            (entry & MAC_48BIT_MASK) == (mac48 & MAC_48BIT_MASK)) {
            return (int)((entry & MAC_ENTRY_PORT_MASK) >> MAC_ENTRY_PORT_SHIFT);
        }

        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
    return -1;
}

/* =========================================================================
 * KERNEL PRINCIPALE — bridge_kernel
 * =========================================================================
 * Dimensioni: <<<n_ports*N_QUEUES_PER_PORT, 32, 0, stream>>>
 * Ogni blocco risolve la propria identità UNA VOLTA (kp.queues[blockIdx.x])
 * e poi resta dedicato a quella coppia (porta, coda) per tutta la vita
 * del kernel persistente.
 */
__global__ void bridge_kernel(struct bridge_kernel_params kp)
{
    /* ── Identità del blocco (letta una volta, in shared memory) ────────── */
    __shared__ struct bridge_queue_ctx ctx;
    if (threadIdx.x == 0)
        ctx = kp.queues[blockIdx.x];
    __syncthreads();

    /* ── Shared memory per la ricezione cooperativa ─────────────────────── */
    __shared__ uint64_t rx_first_pkt_idx;
    __shared__ uint32_t rx_pkt_count;
    __shared__ struct doca_gpu_dev_eth_rxq_attr rx_attr[MAX_RX_NUM_PKTS];

    /* next_wqe_slot[d]/last_wqe_key[d]: come in v1, ma ora indicizzati
     * sulle porte di destinazione di QUESTO blocco (privati per blocco,
     * nessuna contesa con altri blocchi — vedi commento in testa al file). */
    __shared__ uint32_t next_wqe_slot[MAX_N_PORTS];
    __shared__ uint64_t last_wqe_key[MAX_N_PORTS];

    __shared__ uint32_t batch_flood;
    __shared__ uint32_t batch_unicast;
    __shared__ uint32_t batch_drop;

    /* wqe_base[d]: indice assoluto del prossimo WQE su txq_gpu[d] di
     * QUESTO blocco. Deve essere shared (letto da tutti i thread in fase 2). */
    __shared__ uint64_t wqe_base[MAX_N_PORTS];

    /* Solo thread 0, persistenti per tutta la vita del kernel */
    uint64_t cqe_idx[MAX_N_PORTS];

    doca_error_t ret;

    if (threadIdx.x == 0) {
        for (int d = 0; d < kp.n_ports; d++) {
            wqe_base[d] = 0;
            cqe_idx[d]  = 0;
        }
    }
    __syncthreads();

    /* =====================================================================
     * LOOP PRINCIPALE — questo blocco serve SOLO ctx.rxq_gpu
     * ===================================================================== */
    while (DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) == 0) {

        /* ==================================================================
         * FASE 1: RICEZIONE (32 thread cooperano, EXEC_SCOPE_BLOCK)
         * ================================================================== */
        ret = doca_gpu_dev_eth_rxq_recv<
                DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
                DOCA_GPUNETIO_ETH_MCST_AUTO,
                DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
                DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
            ctx.rxq_gpu,
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

        /* ==================================================================
         * FASE 2A: RESET CONTATORI BATCH (solo thread 0)
         * ================================================================== */
        if (threadIdx.x == 0) {
            for (int d = 0; d < kp.n_ports; d++) {
                next_wqe_slot[d] = 0;
                last_wqe_key[d]  = 0;
            }
            batch_flood   = 0;
            batch_unicast = 0;
            batch_drop    = 0;
        }
        __syncthreads();

        /* ==================================================================
         * FASE 2B: BACKWARD LEARNING + FIB LOOKUP + WQE FILL PARALLELO
         * (identica a v1, ma "src" è sempre ctx.port: fisso per il blocco)
         * ================================================================== */
        uint32_t pkt_idx = threadIdx.x;
        while (pkt_idx < rx_pkt_count) {

            uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                ctx.rxq_gpu, rx_first_pkt_idx + pkt_idx);
            const uint8_t *eth = (const uint8_t *)(uintptr_t)pkt_addr;

            uint64_t dst_mac48 = mac_to_u64(eth);
            uint64_t src_mac48 = mac_to_u64(eth + 6);

            mac_learn(kp.mac_table, src_mac48, ctx.port, kp.flap_ring, kp.flap_ring_head);

            uint32_t egress_mask;
            if (is_8021d_reserved(dst_mac48)) {
                egress_mask = 0u;
                atomicAdd(&batch_drop, 1u);
            } else {
                int dst_port = mac_lookup(kp.mac_table, dst_mac48);

                if (dst_port == ctx.port) {
                    egress_mask = 0u;
                    atomicAdd(&batch_drop, 1u);
                } else if (dst_port < 0) {
                    egress_mask = ((1u << kp.n_ports) - 1u) & ~(1u << ctx.port);
                    atomicAdd(&batch_flood, 1u);
                } else {
                    egress_mask = (1u << dst_port);
                    atomicAdd(&batch_unicast, 1u);
                }
            }

            for (int d = 0; d < kp.n_ports; d++) {
                if (!(egress_mask & (1u << d)))
                    continue;

                uint32_t mkey = ctx.rxq_mkey_for_dst[d];
                uint32_t slot = atomicAdd(&next_wqe_slot[d], 1u);

                uint64_t key = ((uint64_t)slot << 32) | (uint64_t)pkt_idx;
                atomicMax((unsigned long long *)&last_wqe_key[d],
                          (unsigned long long)key);

                struct doca_gpu_dev_eth_txq_wqe *wqe =
                    doca_gpu_dev_eth_txq_get_wqe_ptr(
                        ctx.txq_gpu[d], wqe_base[d] + slot);
                doca_gpu_dev_eth_txq_wqe_prepare_send(
                    ctx.txq_gpu[d], wqe, wqe_base[d] + slot,
                    pkt_addr, mkey, rx_attr[pkt_idx].bytes,
                    DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);
            }

            pkt_idx += blockDim.x;
        }

        __syncthreads();

        /* ==================================================================
         * FASE 3: FIXUP NOTIFY + SUBMIT + POLL (solo thread 0)
         * ================================================================== */
        if (threadIdx.x == 0) {
            uint64_t batch_fwd = 0;

            for (int d = 0; d < kp.n_ports; d++) {
                uint32_t n = next_wqe_slot[d];
                if (n == 0)
                    continue;

                uint32_t last_pkt_idx = (uint32_t)(last_wqe_key[d] & 0xFFFFFFFFu);
                uint64_t last_pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                    ctx.rxq_gpu, rx_first_pkt_idx + last_pkt_idx);
                uint32_t last_pkt_bytes = rx_attr[last_pkt_idx].bytes;
                uint32_t last_pkt_mkey  = ctx.rxq_mkey_for_dst[d];

                uint64_t last_slot = wqe_base[d] + n - 1;
                struct doca_gpu_dev_eth_txq_wqe *last_wqe =
                    doca_gpu_dev_eth_txq_get_wqe_ptr(ctx.txq_gpu[d], last_slot);
                doca_gpu_dev_eth_txq_wqe_prepare_send(
                    ctx.txq_gpu[d], last_wqe, last_slot,
                    last_pkt_addr,
                    last_pkt_mkey,
                    last_pkt_bytes,
                    DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY);

                doca_gpu_dev_eth_txq_submit(ctx.txq_gpu[d], wqe_base[d] + n);

                ret = doca_gpu_dev_eth_txq_poll_completion_at<
                        DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
                    ctx.txq_gpu[d], cqe_idx[d],
                    DOCA_GPUNETIO_ETH_WAIT_FLAG_B);

                if (ret != DOCA_SUCCESS) {
                    DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) = 1;
                    break;
                }

                wqe_base[d] += n;
                cqe_idx[d]  += 1;
                batch_fwd   += n;
            }

            /* Pubblica verso la CPU una volta per batch, aggregando con
             * atomicAdd sui contatori globali condivisi da tutti i blocchi.
             * __threadfence_system() rende visibile alla CPU questa scrittura
             * e tutte quelle precedenti (incluso mac_learn sul flap_ring). */
            atomicAdd((unsigned long long *)&kp.rx_pkt_total[ctx.port],
                      (unsigned long long)rx_pkt_count);
            atomicAdd((unsigned long long *)&kp.rx_pkt_per_queue[blockIdx.x],
                      (unsigned long long)rx_pkt_count);
            atomicAdd((unsigned long long *)kp.flood_count,   (unsigned long long)batch_flood);
            atomicAdd((unsigned long long *)kp.unicast_count, (unsigned long long)batch_unicast);
            atomicAdd((unsigned long long *)kp.drop_count,    (unsigned long long)batch_drop);
            atomicAdd((unsigned long long *)kp.fwd_count,     (unsigned long long)batch_fwd);
            __threadfence_system();
        }
        __syncthreads();

    } /* while (!exit_cond) */
}

/* =========================================================================
 * WRAPPER EXTERN "C"
 * ========================================================================= */
extern "C" {

doca_error_t kernel_launch_bridge(cudaStream_t stream,
                                   struct bridge_kernel_params *kp)
{
    cudaError_t cuda_ret;

    if (!kp || !kp->mac_table || !kp->exit_cond || !kp->fwd_count || !kp->queues)
        return DOCA_ERROR_INVALID_VALUE;

    if (!kp->rx_pkt_total || !kp->rx_pkt_per_queue || !kp->flood_count ||
        !kp->unicast_count || !kp->drop_count || !kp->flap_ring || !kp->flap_ring_head)
        return DOCA_ERROR_INVALID_VALUE;

    if (kp->n_ports < 2 || kp->n_ports > MAX_N_PORTS)
        return DOCA_ERROR_INVALID_VALUE;

    if (kp->n_queues != N_QUEUES_PER_PORT)
        return DOCA_ERROR_INVALID_VALUE;

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA error before kernel launch: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    int n_blocks = kp->n_ports * kp->n_queues;
    bridge_kernel<<<n_blocks, BRIDGE_BLOCK_THREADS, 0, stream>>>(*kp);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern "C" */
