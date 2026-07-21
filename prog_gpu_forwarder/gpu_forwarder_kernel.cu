/*
 * gpu_forwarder_kernel.cu — CUDA kernel for zero-copy GPU UDP forwarding.
 *
 * All 32 threads in the block cooperate to receive a batch of packets into
 * the RXQ cyclic GPU buffer.  Each thread then modifies its packet's UDP
 * destination port in-place and fills a TXQ WQE that points directly to
 * the same GPU memory (zero-copy: the NIC DMA-reads from GPU to transmit).
 * Thread 0 rings the doorbell and waits for the completion CQE.
 *
 * Packet header layout assumed (standard Ethernet/IPv4/UDP):
 *   [0..13]  Ethernet (14 B)
 *   [14..33] IPv4     (20 B)
 *   [34..35] UDP src port
 *   [36..37] UDP dst port  ← modified here
 *   [38..39] UDP length
 *   [40..41] UDP checksum  ← zeroed (NIC recalculates via L4 offload)
 */

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gpu_forwarder.h"

#define UDP_DST_PORT_U16  18   /* byte offset 36  / 2 */
#define UDP_CHECKSUM_U16  20   /* byte offset 40  / 2 */
#define UDP_MIN_PKT_LEN   42   /* need at least this many bytes to touch checksum */

/*
 * MAC addresses packed as three uint16_t values each (network byte order).
 * Ethernet frame layout: [dst_mac 6B][src_mac 6B][ethertype 2B]...
 *   dst_mac_be[0..2] → bytes  0..5
 *   src_mac_be[0..2] → bytes  6..11
 * Pass {0,0,0} for either to leave the original unchanged.
 */
__global__ void forward_packets_kernel(
    struct doca_gpu_eth_rxq *rxq,
    struct doca_gpu_eth_txq *txq,
    uint32_t rxq_mkey,
    uint16_t new_dst_port_be,
    uint16_t dst_mac_be0, uint16_t dst_mac_be1, uint16_t dst_mac_be2,
    uint16_t src_mac_be0, uint16_t src_mac_be1, uint16_t src_mac_be2,
    uint32_t *exit_cond,
    uint64_t *fwd_count)
{
    __shared__ uint64_t                     out_first_pkt_idx;
    __shared__ uint32_t                     out_pkt_num;
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr[MAX_RX_NUM_PKTS];
    __shared__ uint64_t                     wqe_base;

    uint64_t cqe_idx = 0;   /* monotonically increasing CQE sequence (thread 0 only) */
    uint64_t tot_fwd = 0;   /* total forwarded packets       (thread 0 only) */
    doca_error_t ret;

    if (threadIdx.x == 0)
        wqe_base = 0;
    __syncthreads();

    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {

        /* ---- RECEIVE -------------------------------------------------- */
        /* All 32 threads cooperate; returns when ≥1 packet arrives or timeout. */
        ret = doca_gpu_dev_eth_rxq_recv<
            DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
            DOCA_GPUNETIO_ETH_MCST_AUTO,
            DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
            DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
            rxq,
            MAX_RX_NUM_PKTS,
            MAX_RX_TIMEOUT_NS,
            &out_first_pkt_idx,
            &out_pkt_num,
            out_attr);

        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0)
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            break;
        }

        if (out_pkt_num == 0)
            continue;

        /* ---- MODIFY + FILL WQEs --------------------------------------- */
        /*
         * Round-robin across threads: thread t handles packets at indices
         * t, t+blockDim.x, t+2*blockDim.x, …
         * Only the packet at buf_idx == out_pkt_num-1 sets NOTIFY so the
         * NIC generates exactly one CQE per submitted batch.
         */
        uint32_t buf_idx = threadIdx.x;
        while (buf_idx < out_pkt_num) {
            uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                rxq, out_first_pkt_idx + buf_idx);

            if (out_attr[buf_idx].bytes >= UDP_MIN_PKT_LEN) {
                uint16_t *p = (uint16_t *)(uintptr_t)pkt_addr;

                /* Ethernet dst MAC: bytes 0..5 = p[0], p[1], p[2] */
                if (dst_mac_be0 | dst_mac_be1 | dst_mac_be2) {
                    p[0] = dst_mac_be0;
                    p[1] = dst_mac_be1;
                    p[2] = dst_mac_be2;
                }
                /* Ethernet src MAC: bytes 6..11 = p[3], p[4], p[5] */
                if (src_mac_be0 | src_mac_be1 | src_mac_be2) {
                    p[3] = src_mac_be0;
                    p[4] = src_mac_be1;
                    p[5] = src_mac_be2;
                }

                /* Rewrite UDP destination port (byte offset 36 = p[18]). */
                p[UDP_DST_PORT_U16] = new_dst_port_be;
                /* Zero UDP checksum (byte offset 40 = p[20]); NIC recalculates. */
                p[UDP_CHECKSUM_U16] = 0;
            }

            enum doca_gpu_eth_send_flags flags =
                (buf_idx == out_pkt_num - 1) ?
                    DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY :
                    DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;

            /* Zero-copy WQE: NIC will DMA-read from RXQ GPU buffer. */
            struct doca_gpu_dev_eth_txq_wqe *wqe =
                doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_base + buf_idx);
            doca_gpu_dev_eth_txq_wqe_prepare_send(
                txq, wqe, wqe_base + buf_idx,
                pkt_addr, rxq_mkey,
                out_attr[buf_idx].bytes, flags);

            buf_idx += blockDim.x;
        }
        __syncthreads();   /* all WQEs written to GPU memory */

        /* ---- SUBMIT + POLL -------------------------------------------- */
        if (threadIdx.x == 0) {
            /* Ring the doorbell for all WQEs in this batch. */
            doca_gpu_dev_eth_txq_submit(txq, wqe_base + out_pkt_num);

            /* Wait for the CQE triggered by the NOTIFY WQE. */
            ret = doca_gpu_dev_eth_txq_poll_completion_at<
                DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
                DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
                txq, cqe_idx, DOCA_GPUNETIO_ETH_WAIT_FLAG_B);

            if (ret != DOCA_SUCCESS)
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;

            cqe_idx++;
            wqe_base += out_pkt_num;
            tot_fwd  += out_pkt_num;
        }
        __syncthreads();   /* wqe_base updated; safe to start next recv */
    }

    if (threadIdx.x == 0) {
        *fwd_count = tot_fwd;
        __threadfence_system();   /* make visible to CPU after stream sync */
    }
}

/* ---- Host-callable wrapper ------------------------------------------- */

extern "C" {

doca_error_t kernel_forward_packets(
    cudaStream_t      stream,
    struct rxq_queue *rxq,
    struct txq_queue *txq,
    uint32_t          rxq_mkey,
    uint16_t          new_dst_port_be,
    const uint8_t    *dst_mac,     /* 6 bytes, network order; NULL = don't rewrite */
    const uint8_t    *src_mac,     /* 6 bytes, network order; NULL = don't rewrite */
    uint32_t         *gpu_exit_cond,
    uint64_t         *gpu_fwd_count)
{
    cudaError_t res;

    if (!rxq || !txq || !gpu_exit_cond || !gpu_fwd_count)
        return DOCA_ERROR_INVALID_VALUE;

    res = cudaGetLastError();
    if (res != cudaSuccess)
        return DOCA_ERROR_BAD_STATE;

    /* Pack each MAC as three uint16_t in network byte order. */
    auto mac_w = [](const uint8_t *m, int i) -> uint16_t {
        return m ? (uint16_t)((m[i*2] << 8) | m[i*2+1]) : 0;
    };

    forward_packets_kernel<<<1, CUDA_BLOCK_THREADS, 0, stream>>>(
        rxq->eth_rxq_gpu,
        txq->eth_txq_gpu,
        rxq_mkey,
        new_dst_port_be,
        mac_w(dst_mac,0), mac_w(dst_mac,1), mac_w(dst_mac,2),
        mac_w(src_mac,0), mac_w(src_mac,1), mac_w(src_mac,2),
        gpu_exit_cond,
        gpu_fwd_count);

    res = cudaGetLastError();
    if (res != cudaSuccess)
        return DOCA_ERROR_BAD_STATE;

    return DOCA_SUCCESS;
}

} /* extern "C" */
