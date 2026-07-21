/*
 * gpu_forwarder.h — shared types and constants for GPU zero-copy UDP forwarder.
 *
 * Data path: NIC DMA → GPU RXQ cyclic buffer → CUDA kernel modifies UDP dst
 * port in-place → CUDA kernel writes TXQ WQEs pointing to the same GPU memory
 * (zero-copy) → NIC reads and transmits.  No CPU involvement per packet.
 */
#ifndef GPU_FORWARDER_H
#define GPU_FORWARDER_H

#include <stdint.h>
#include <cuda_runtime.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_flow.h>

/* RXQ cyclic buffer sizing */
#define MAX_PKT_NUM         16384U
#define MAX_PKT_SIZE        2048U

/* Per-recv-call limits */
#define MAX_RX_NUM_PKTS     2048U
#define MAX_RX_TIMEOUT_NS   500000ULL

/* TXQ descriptor ring size */
#define MAX_SQ_DESCR_NUM    8192U

/* CUDA block size (1 warp) */
#define CUDA_BLOCK_THREADS  32U

/* DOCA Flow */
#define FLOW_NB_COUNTERS    524228U

/* Round up size to alignment boundary (modifies size in-place) */
#define ALIGN_SIZE(sz, al)  ((sz) = (((sz) + (al) - 1) / (al)) * (al))

/* GPU RXQ state */
struct rxq_queue {
    struct doca_gpu             *gpu_dev;
    struct doca_dev             *ddev;
    struct doca_ctx             *eth_rxq_ctx;
    struct doca_eth_rxq         *eth_rxq_cpu;
    struct doca_gpu_eth_rxq     *eth_rxq_gpu;
    struct doca_mmap            *pkt_buff_mmap;
    void                        *gpu_pkt_addr;
    int                          dmabuf_fd;
    struct doca_flow_port       *port;
    struct doca_flow_pipe       *rxq_pipe;
    struct doca_flow_pipe       *root_pipe;
    struct doca_flow_pipe_entry *root_udp_entry;
    uint32_t                     rxq_mkey;   /* network-byte-order mkey for zero-copy TX */
};

/* GPU TXQ state (no packet buffer — zero-copy from RXQ memory) */
struct txq_queue {
    struct doca_gpu             *gpu_dev;
    struct doca_dev             *ddev;
    struct doca_ctx             *eth_txq_ctx;
    struct doca_eth_txq         *eth_txq_cpu;
    struct doca_gpu_eth_txq     *eth_txq_gpu;
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Launch the GPU forwarder kernel on @stream.
 * The kernel runs until *gpu_exit_cond != 0.
 * On exit writes total forwarded packets to *gpu_fwd_count.
 *
 * @rxq_mkey       already in network byte order (htobe32 applied on host)
 * @new_dst_port_be  already in network byte order (htons applied on host)
 */
doca_error_t kernel_forward_packets(
    cudaStream_t      stream,
    struct rxq_queue *rxq,
    struct txq_queue *txq,
    uint32_t          rxq_mkey,
    uint16_t          new_dst_port_be,
    const uint8_t    *dst_mac,       /* 6 bytes, network order; NULL = leave unchanged */
    const uint8_t    *src_mac,       /* 6 bytes, network order; NULL = leave unchanged */
    uint32_t         *gpu_exit_cond,
    uint64_t         *gpu_fwd_count);

#ifdef __cplusplus
}
#endif

#endif /* GPU_FORWARDER_H */
