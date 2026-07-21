/*
 * gpu_forwarder.c — host-side setup for GPU zero-copy UDP forwarder.
 *
 * Usage:
 *   gpu_forwarder -n <nic_pci> -g <gpu_pci> [-i <cuda_id>] [-r <rx_port>] [-t <tx_port>]
 *
 * Example:
 *   gpu_forwarder -n 03:00.0 -g 01:00.0 -r 5001 -t 5002
 *
 * Packets arriving at the BlueField-2 NIC on UDP dst port <rx_port> are
 * DMA'd directly into GPU memory, the destination port is rewritten to
 * <tx_port> by the CUDA kernel, and the packet is sent back out from GPU
 * memory without any CPU involvement per packet.
 *
 * NOTE: Only the UDP destination port and checksum are modified.
 *       The Ethernet/IP source and destination addresses are preserved
 *       from the original received packet.  Adjust the kernel if your
 *       test topology requires different MAC/IP rewriting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <arpa/inet.h>

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

#include "gpu_forwarder.h"

#define DEFAULT_RX_PORT  5001
#define DEFAULT_TX_PORT  5002

static volatile bool g_force_quit = false;

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
        DOCA_GPUNETIO_VOLATILE(g_force_quit) = true;
}

static size_t host_page_size(void)
{
    long ret = sysconf(_SC_PAGESIZE);
    return (ret <= 0) ? 4096UL : (size_t)ret;
}

/* ---- DOCA device -------------------------------------------------------- */

static doca_error_t open_device_by_pci(const char *pci_addr, struct doca_dev **dev)
{
    struct doca_devinfo **list;
    uint32_t n, i;
    uint8_t is_equal;
    doca_error_t res;

    res = doca_devinfo_create_list(&list, &n);
    if (res != DOCA_SUCCESS) return res;

    for (i = 0; i < n; i++) {
        res = doca_devinfo_is_equal_pci_addr(list[i], pci_addr, &is_equal);
        if (res == DOCA_SUCCESS && is_equal) {
            res = doca_dev_open(list[i], dev);
            doca_devinfo_destroy_list(list);
            return res;
        }
    }
    doca_devinfo_destroy_list(list);
    fprintf(stderr, "NIC device %s not found\n", pci_addr);
    return DOCA_ERROR_NOT_FOUND;
}

/* ---- DOCA Flow ---------------------------------------------------------- */

static struct doca_flow_port *df_port;

static doca_error_t flow_init(void)
{
    struct doca_flow_cfg *cfg;
    doca_error_t res;

    res = doca_flow_cfg_create(&cfg);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_cfg_set_pipe_queues(cfg, 1);
    doca_flow_cfg_set_mode_args(cfg, "vnf");
    doca_flow_cfg_set_nr_counters(cfg, FLOW_NB_COUNTERS);

    res = doca_flow_init(cfg);
    doca_flow_cfg_destroy(cfg);
    return res;
}

static doca_error_t flow_start_port(struct doca_dev *dev)
{
    struct doca_flow_port_cfg *port_cfg;
    doca_error_t res;

    res = doca_flow_port_cfg_create(&port_cfg);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_port_cfg_set_port_id(port_cfg, 0);
    doca_flow_port_cfg_set_dev(port_cfg, dev);

    res = doca_flow_port_start(port_cfg, &df_port);
    doca_flow_port_cfg_destroy(port_cfg);
    return res;
}

/*
 * BASIC pipe: matches IPv4+UDP metadata, RSS-hashes to queue 0.
 * All IPv4+UDP traffic that reaches this pipe lands on the GPU RXQ.
 */
static doca_error_t create_udp_pipe(struct rxq_queue *rxq)
{
    struct doca_flow_match match  = {0};
    struct doca_flow_fwd   fwd   = {0};
    struct doca_flow_fwd   miss  = {0};
    struct doca_flow_monitor mon  = { .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED };
    struct doca_flow_pipe_cfg *pipe_cfg;
    struct doca_flow_pipe_entry *entry;
    uint16_t rss_q[1] = {0};
    doca_error_t res;

    match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
    match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;

    doca_eth_rxq_apply_queue_id(rxq->eth_rxq_cpu, 0);

    fwd.type               = DOCA_FLOW_FWD_RSS;
    fwd.rss_type           = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array   = rss_q;
    fwd.rss.outer_flags    = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
    fwd.rss.nr_queues      = 1;
    miss.type              = DOCA_FLOW_FWD_DROP;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,   "GPU_UDP_PIPE");
    doca_flow_pipe_cfg_set_type(pipe_cfg,   DOCA_FLOW_PIPE_BASIC);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, false);
    doca_flow_pipe_cfg_set_match(pipe_cfg,  &match, NULL);
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, &fwd, &miss, &rxq->rxq_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) return res;

    res = doca_flow_pipe_basic_add_entry(0, rxq->rxq_pipe, &match, 0,
                                          NULL, NULL, NULL,
                                          DOCA_FLOW_ENTRY_FLAGS_NO_WAIT,
                                          NULL, &entry);
    if (res != DOCA_SUCCESS) return res;

    return doca_flow_entries_process(df_port, 0, 0, 0);
}

/*
 * CONTROL root pipe: routes IPv4+UDP with dst_port==rx_port to the UDP pipe.
 * All other traffic is dropped by the CONTROL pipe's default behaviour.
 */
static doca_error_t create_root_pipe(struct rxq_queue *rxq, uint16_t rx_port)
{
    struct doca_flow_match udp_match = {
        .outer.eth.type         = htons(DOCA_FLOW_ETHER_TYPE_IPV4),
        .outer.l3_type          = DOCA_FLOW_L3_TYPE_IP4,
        .outer.ip4.next_proto   = IPPROTO_UDP,
        .outer.udp.l4_port.dst_port = htons(rx_port),
    };
    struct doca_flow_fwd fwd = {
        .type      = DOCA_FLOW_FWD_PIPE,
        .next_pipe = NULL,   /* filled below */
    };
    struct doca_flow_monitor mon = { .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED };
    struct doca_flow_pipe_cfg *pipe_cfg;
    doca_error_t res;

    fwd.next_pipe = rxq->rxq_pipe;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,    "ROOT_PIPE");
    doca_flow_pipe_cfg_set_type(pipe_cfg,    DOCA_FLOW_PIPE_CONTROL);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, NULL, NULL, &rxq->root_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) return res;

    res = doca_flow_pipe_control_add_entry(
        0,                  /* pipe_queue */
        rxq->root_pipe,
        &udp_match,
        NULL,               /* match_mask */
        NULL, NULL, NULL,   /* actions, actions_mask, actions_descs */
        NULL,               /* actions_descs_mask */
        NULL,               /* monitor */
        0,                  /* priority */
        &fwd,
        NULL,               /* timeout_entry */
        &rxq->root_udp_entry);
    if (res != DOCA_SUCCESS) return res;

    return doca_flow_entries_process(df_port, 0, 0, 0);
}

/* ---- RXQ ---------------------------------------------------------------- */

static doca_error_t create_rxq(struct rxq_queue *rxq, struct doca_gpu *gpu_dev,
                                 int cuda_id, struct doca_dev *ddev,
                                 uint16_t rx_port)
{
    doca_error_t res;
    uint32_t cyclic_buf_size = 0;
    struct cudaDeviceProp prop;

    rxq->gpu_dev = gpu_dev;
    rxq->ddev    = ddev;
    rxq->port    = df_port;

    res = doca_eth_rxq_create(ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &rxq->eth_rxq_cpu);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_rxq_create: %s\n", doca_error_get_descr(res)); return res; }

    res = doca_eth_rxq_set_type(rxq->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_rxq_set_type: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC,
                                                 0, 0, MAX_PKT_SIZE, MAX_PKT_NUM,
                                                 0, 0, 0, &cyclic_buf_size);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "estimate_pkt_buf_size: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_mmap_create(&rxq->pkt_buff_mmap);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_mmap_create: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_mmap_add_dev(rxq->pkt_buff_mmap, ddev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_mmap_add_dev: %s\n", doca_error_get_descr(res)); goto err; }

    ALIGN_SIZE(cyclic_buf_size, host_page_size());

    res = doca_gpu_mem_alloc(gpu_dev, cyclic_buf_size, host_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU,
                              &rxq->gpu_pkt_addr, NULL);
    if (res != DOCA_SUCCESS || !rxq->gpu_pkt_addr) {
        fprintf(stderr, "doca_gpu_mem_alloc RXQ: %s\n", doca_error_get_descr(res));
        goto err;
    }

    /* Try DMABuf first; fall back to nvidia-peermem. */
    res = doca_gpu_dmabuf_fd(gpu_dev, rxq->gpu_pkt_addr, cyclic_buf_size, &rxq->dmabuf_fd);
    if (res != DOCA_SUCCESS) {
        printf("DMABuf not available, using nvidia-peermem\n");
        res = doca_mmap_set_memrange(rxq->pkt_buff_mmap, rxq->gpu_pkt_addr, cyclic_buf_size);
    } else {
        res = doca_mmap_set_dmabuf_memrange(rxq->pkt_buff_mmap, rxq->dmabuf_fd,
                                             rxq->gpu_pkt_addr, 0, cyclic_buf_size);
    }
    if (res != DOCA_SUCCESS) { fprintf(stderr, "mmap set_memrange: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_mmap_set_permissions(rxq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "mmap set_permissions: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_mmap_start(rxq->pkt_buff_mmap);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_mmap_start: %s\n", doca_error_get_descr(res)); goto err; }

    /*
     * Retrieve the hardware memory key for this GPU buffer.
     * The CUDA kernel passes this mkey in TXQ WQEs so the NIC can
     * DMA-read directly from GPU memory (zero-copy send).
     */
    res = doca_mmap_get_mkey(rxq->pkt_buff_mmap, ddev, &rxq->rxq_mkey);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_mmap_get_mkey: %s\n", doca_error_get_descr(res)); goto err; }
    rxq->rxq_mkey = htobe32(rxq->rxq_mkey);   /* NIC expects network byte order */

    res = doca_eth_rxq_set_pkt_buf(rxq->eth_rxq_cpu, rxq->pkt_buff_mmap, 0, cyclic_buf_size);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_rxq_set_pkt_buf: %s\n", doca_error_get_descr(res)); goto err; }

    /* Pre-Hopper GPUs need the multicast-QP workaround. */
    cudaGetDeviceProperties(&prop, cuda_id);
    if (prop.major < 9) {
        res = doca_eth_rxq_gpu_enable_mcst_qp(rxq->eth_rxq_cpu);
        if (res != DOCA_SUCCESS) { fprintf(stderr, "enable_mcst_qp: %s\n", doca_error_get_descr(res)); goto err; }
    }

    rxq->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(rxq->eth_rxq_cpu);
    if (!rxq->eth_rxq_ctx) { fprintf(stderr, "doca_eth_rxq_as_doca_ctx failed\n"); goto err; }

    res = doca_ctx_set_datapath_on_gpu(rxq->eth_rxq_ctx, gpu_dev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "ctx_set_datapath_on_gpu RXQ: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_ctx_start(rxq->eth_rxq_ctx);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_ctx_start RXQ: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_eth_rxq_get_gpu_handle(rxq->eth_rxq_cpu, &rxq->eth_rxq_gpu);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_rxq_get_gpu_handle: %s\n", doca_error_get_descr(res)); goto err; }

    res = create_udp_pipe(rxq);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "create_udp_pipe: %s\n", doca_error_get_descr(res)); goto err; }

    res = create_root_pipe(rxq, rx_port);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "create_root_pipe: %s\n", doca_error_get_descr(res)); goto err; }

    printf("RXQ ready — GPU buffer %p  mkey 0x%08x  rx_port %u\n",
           rxq->gpu_pkt_addr, rxq->rxq_mkey, rx_port);
    return DOCA_SUCCESS;

err:
    if (rxq->eth_rxq_ctx)   doca_ctx_stop(rxq->eth_rxq_ctx);
    if (rxq->gpu_pkt_addr)  doca_gpu_mem_free(gpu_dev, rxq->gpu_pkt_addr);
    if (rxq->eth_rxq_cpu)   doca_eth_rxq_destroy(rxq->eth_rxq_cpu);
    if (rxq->pkt_buff_mmap) doca_mmap_destroy(rxq->pkt_buff_mmap);
    return DOCA_ERROR_BAD_STATE;
}

/* ---- TXQ ---------------------------------------------------------------- */

/*
 * Create a GPU TXQ with no pre-allocated packet buffer.
 * The kernel writes WQEs that reference RXQ GPU memory directly (zero-copy).
 */
static doca_error_t create_txq(struct txq_queue *txq, struct doca_gpu *gpu_dev,
                                 struct doca_dev *ddev)
{
    doca_error_t res;

    txq->gpu_dev = gpu_dev;
    txq->ddev    = ddev;

    res = doca_eth_txq_create(ddev, MAX_SQ_DESCR_NUM, &txq->eth_txq_cpu);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_txq_create: %s\n", doca_error_get_descr(res)); return res; }

    /* Enable L3/L4 checksum offload: NIC recalculates IP/UDP checksums. */
    res = doca_eth_txq_set_l3_chksum_offload(txq->eth_txq_cpu, 1);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "set_l3_chksum_offload: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_eth_txq_set_l4_chksum_offload(txq->eth_txq_cpu, 1);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "set_l4_chksum_offload: %s\n", doca_error_get_descr(res)); goto err; }

    /* CQE polling is done by the GPU kernel, not the CPU. */
    res = doca_eth_txq_gpu_set_completion_on_gpu(txq->eth_txq_cpu);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "set_completion_on_gpu: %s\n", doca_error_get_descr(res)); goto err; }

    txq->eth_txq_ctx = doca_eth_txq_as_doca_ctx(txq->eth_txq_cpu);
    if (!txq->eth_txq_ctx) { fprintf(stderr, "doca_eth_txq_as_doca_ctx failed\n"); goto err; }

    res = doca_ctx_set_datapath_on_gpu(txq->eth_txq_ctx, gpu_dev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "ctx_set_datapath_on_gpu TXQ: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_ctx_start(txq->eth_txq_ctx);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_ctx_start TXQ: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_eth_txq_apply_queue_id(txq->eth_txq_cpu, 0);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "txq_apply_queue_id: %s\n", doca_error_get_descr(res)); goto err; }

    res = doca_eth_txq_get_gpu_handle(txq->eth_txq_cpu, &txq->eth_txq_gpu);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_eth_txq_get_gpu_handle: %s\n", doca_error_get_descr(res)); goto err; }

    printf("TXQ ready — %u descriptors\n", MAX_SQ_DESCR_NUM);
    return DOCA_SUCCESS;

err:
    if (txq->eth_txq_ctx) doca_ctx_stop(txq->eth_txq_ctx);
    if (txq->eth_txq_cpu) doca_eth_txq_destroy(txq->eth_txq_cpu);
    return DOCA_ERROR_BAD_STATE;
}

/* ---- Usage / main ------------------------------------------------------- */

static int parse_mac(const char *str, uint8_t mac[6])
{
    unsigned int v[6];
    if (sscanf(str, "%x:%x:%x:%x:%x:%x",
               &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6)
        return -1;
    for (int i = 0; i < 6; i++) mac[i] = (uint8_t)v[i];
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -n <nic_pci> -g <gpu_pci> [-i <cuda_id>] [-r <rx_port>] [-t <tx_port>]\n"
        "       [-d <dst_mac>] [-s <src_mac>]\n"
        "  -n  BlueField-2 NIC PCIe address  (e.g. 03:00.0)\n"
        "  -g  GPU PCIe address              (e.g. 01:00.0)\n"
        "  -i  CUDA device index             (default: 0)\n"
        "  -r  UDP receive port              (default: %d)\n"
        "  -t  UDP transmit (forwarded) port (default: %d)\n"
        "  -d  Ethernet destination MAC to write into forwarded packets\n"
        "      (e.g. aa:bb:cc:dd:ee:ff — set to MAC of receiving NIC)\n"
        "  -s  Ethernet source MAC to write into forwarded packets\n"
        "      (e.g. aa:bb:cc:dd:ee:ff — typically BF2 port MAC)\n",
        prog, DEFAULT_RX_PORT, DEFAULT_TX_PORT);
}

int main(int argc, char *argv[])
{
    char     nic_pci[DOCA_DEVINFO_PCI_ADDR_SIZE] = {0};
    char     gpu_pci[64] = {0};
    int      cuda_id  = 0;
    uint16_t rx_port  = DEFAULT_RX_PORT;
    uint16_t tx_port  = DEFAULT_TX_PORT;
    uint8_t  dst_mac[6] = {0};   /* all-zero = don't rewrite */
    uint8_t  src_mac[6] = {0};
    int      opt;

    while ((opt = getopt(argc, argv, "n:g:i:r:t:d:s:h")) != -1) {
        switch (opt) {
        case 'n': strncpy(nic_pci, optarg, sizeof(nic_pci) - 1); break;
        case 'g': strncpy(gpu_pci, optarg, sizeof(gpu_pci) - 1); break;
        case 'i': cuda_id = atoi(optarg);              break;
        case 'r': rx_port = (uint16_t)atoi(optarg);   break;
        case 't': tx_port = (uint16_t)atoi(optarg);   break;
        case 'd':
            if (parse_mac(optarg, dst_mac) != 0) {
                fprintf(stderr, "Formato MAC non valido: %s\n", optarg);
                return 1;
            }
            break;
        case 's':
            if (parse_mac(optarg, src_mac) != 0) {
                fprintf(stderr, "Formato MAC non valido: %s\n", optarg);
                return 1;
            }
            break;
        default:  usage(argv[0]); return 1;
        }
    }
    if (!nic_pci[0] || !gpu_pci[0]) {
        fprintf(stderr, "Error: -n and -g are required.\n");
        usage(argv[0]);
        return 1;
    }

    doca_error_t  res      = DOCA_SUCCESS;
    struct doca_dev *ddev  = NULL;
    struct doca_gpu *gdev  = NULL;
    struct rxq_queue rxq   = {0};
    struct txq_queue txq   = {0};
    cudaStream_t stream    = NULL;

    uint32_t *gpu_exit = NULL, *cpu_exit = NULL;
    uint64_t *gpu_fwd  = NULL, *cpu_fwd  = NULL;

    cudaSetDevice(cuda_id);

    /* Open NIC device */
    res = open_device_by_pci(nic_pci, &ddev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "open NIC: %s\n", doca_error_get_descr(res)); return 1; }

    /* Init DOCA Flow */
    res = flow_init();
    if (res != DOCA_SUCCESS) { fprintf(stderr, "flow_init: %s\n", doca_error_get_descr(res)); goto cleanup; }

    res = flow_start_port(ddev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "flow_start_port: %s\n", doca_error_get_descr(res)); goto cleanup; }

    /* Create GPU context */
    res = doca_gpu_create(gpu_pci, &gdev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "doca_gpu_create: %s\n", doca_error_get_descr(res)); goto cleanup; }

    /* Create RXQ (includes DOCA Flow pipes for rx_port steering) */
    res = create_rxq(&rxq, gdev, cuda_id, ddev, rx_port);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "create_rxq: %s\n", doca_error_get_descr(res)); goto cleanup; }

    /* Create TXQ (no packet buffer — zero-copy from RXQ memory) */
    res = create_txq(&txq, gdev, ddev);
    if (res != DOCA_SUCCESS) { fprintf(stderr, "create_txq: %s\n", doca_error_get_descr(res)); goto cleanup; }

    /*
     * Exit condition: GPU_CPU so the GPU kernel polls it fast (GPU-side memory)
     * and the CPU can write it to signal stop.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), host_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit, (void **)&cpu_exit);
    if (res != DOCA_SUCCESS || !gpu_exit || !cpu_exit) {
        fprintf(stderr, "alloc exit_cond: %s\n", doca_error_get_descr(res)); goto cleanup;
    }
    cpu_exit[0] = 0;

    /*
     * Forward count: CPU_GPU so the CPU can read it efficiently after sync.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), host_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_fwd, (void **)&cpu_fwd);
    if (res != DOCA_SUCCESS || !gpu_fwd || !cpu_fwd) {
        fprintf(stderr, "alloc fwd_count: %s\n", doca_error_get_descr(res)); goto cleanup;
    }
    cpu_fwd[0] = 0;

    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed\n"); goto cleanup;
    }

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    /* Pass NULL for MAC arrays if all-zero (= don't rewrite). */
    const uint8_t *pdst = (dst_mac[0]|dst_mac[1]|dst_mac[2]|
                            dst_mac[3]|dst_mac[4]|dst_mac[5]) ? dst_mac : NULL;
    const uint8_t *psrc = (src_mac[0]|src_mac[1]|src_mac[2]|
                            src_mac[3]|src_mac[4]|src_mac[5]) ? src_mac : NULL;

    printf("\nGPU forwarder running:\n");
    printf("  NIC:  %s\n", nic_pci);
    printf("  GPU:  %s  (CUDA device %d)\n", gpu_pci, cuda_id);
    printf("  UDP:  port %u  →  port %u\n", rx_port, tx_port);
    if (pdst)
        printf("  dst MAC rewrite: %02x:%02x:%02x:%02x:%02x:%02x\n",
               dst_mac[0], dst_mac[1], dst_mac[2],
               dst_mac[3], dst_mac[4], dst_mac[5]);
    if (psrc)
        printf("  src MAC rewrite: %02x:%02x:%02x:%02x:%02x:%02x\n",
               src_mac[0], src_mac[1], src_mac[2],
               src_mac[3], src_mac[4], src_mac[5]);
    printf("Press Ctrl+C to stop.\n\n");

    res = kernel_forward_packets(stream, &rxq, &txq,
                                  rxq.rxq_mkey, htons(tx_port),
                                  pdst, psrc,
                                  gpu_exit, gpu_fwd);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "kernel_forward_packets: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }

    /* Wait until Ctrl+C */
    while (!DOCA_GPUNETIO_VOLATILE(g_force_quit))
        ;

    /* Signal kernel to exit, then wait for it. */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit) = 1;
    cudaStreamSynchronize(stream);

    printf("\nStopped.  Total forwarded packets: %lu\n", cpu_fwd[0]);

cleanup:
    if (stream)    cudaStreamDestroy(stream);
    if (gpu_exit)  doca_gpu_mem_free(gdev, gpu_exit);
    if (gpu_fwd)   doca_gpu_mem_free(gdev, gpu_fwd);

    /* Stop TXQ before closing device */
    if (txq.eth_txq_ctx) doca_ctx_stop(txq.eth_txq_ctx);
    if (txq.eth_txq_cpu) doca_eth_txq_destroy(txq.eth_txq_cpu);

    /* Tear down RXQ and Flow */
    if (rxq.root_pipe)    doca_flow_pipe_destroy(rxq.root_pipe);
    if (rxq.rxq_pipe)     doca_flow_pipe_destroy(rxq.rxq_pipe);
    if (df_port)          doca_flow_port_stop(df_port);
    if (rxq.eth_rxq_ctx)  doca_ctx_stop(rxq.eth_rxq_ctx);
    if (rxq.gpu_pkt_addr) doca_gpu_mem_free(gdev, rxq.gpu_pkt_addr);
    if (rxq.eth_rxq_cpu)  doca_eth_rxq_destroy(rxq.eth_rxq_cpu);
    if (rxq.pkt_buff_mmap) doca_mmap_destroy(rxq.pkt_buff_mmap);

    if (ddev)  doca_dev_close(ddev);
    if (gdev)  doca_gpu_destroy(gdev);
    if (df_port) doca_flow_destroy();

    printf("Done.\n");
    return (res == DOCA_SUCCESS) ? 0 : 1;
}
