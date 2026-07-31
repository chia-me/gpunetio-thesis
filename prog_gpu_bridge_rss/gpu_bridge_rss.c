/*
 * gpu_bridge_rss.c — setup lato CPU per il GPU L2 Bridge a N porte con
 * RSS multi-coda (v2).
 *
 * Differenze principali rispetto a prog_gpu_bridge (v1):
 *   - Ogni porta ha N_QUEUES_PER_PORT code RX, bilanciate dall'hardware
 *     tramite RSS (hash su IPv4/IPv6/UDP/TCP, vedi RSS_HASH_FIELDS).
 *   - Ogni coppia (porta sorgente, coda) ha un set PRIVATO di TXQ verso
 *     ogni altra porta di destinazione — nessuna TXQ è mai condivisa tra
 *     due blocchi CUDA diversi (vedi gpu_bridge_rss.h e il kernel per il
 *     perché questo evita ogni sincronizzazione cross-block).
 *   - Il kernel CUDA gira come griglia di n_ports*N_QUEUES_PER_PORT blocchi
 *     invece di 1 blocco solo (vedi gpu_bridge_rss_kernel.cu).
 *
 * Utilizzo:
 *   sudo ip netns exec bf2 ./gpu_bridge_rss \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <endian.h>

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

#include "gpu_bridge_rss.h"

/* ── Segnale Ctrl+C ─────────────────────────────────────────────────────── */
static volatile bool g_force_quit = false;

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
        DOCA_GPUNETIO_VOLATILE(g_force_quit) = true;
}

/* ── Utility ────────────────────────────────────────────────────────────── */
static size_t system_page_size(void)
{
    long ret = sysconf(_SC_PAGESIZE);
    return (ret <= 0) ? 4096UL : (size_t)ret;
}

static const char *mac48_to_str(uint64_t mac48)
{
    static char buf[18];
    snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x",
             (unsigned)((mac48 >> 40) & 0xFF), (unsigned)((mac48 >> 32) & 0xFF),
             (unsigned)((mac48 >> 24) & 0xFF), (unsigned)((mac48 >> 16) & 0xFF),
             (unsigned)((mac48 >>  8) & 0xFF), (unsigned)( mac48        & 0xFF));
    return buf;
}

/* ==========================================================================
 * BUDGET CODE HARDWARE
 * ==========================================================================
 * Ogni NIC fisica d ospita:
 *   - N_QUEUES_PER_PORT RX queue (le proprie, bilanciate da RSS)
 *   - N_QUEUES_PER_PORT * (n_ports - 1) TX queue (una per ogni blocco di
 *     ogni ALTRA porta che può forwardare verso d — vedi header)
 * Le NIC BF2/BF3 supportano tipicamente fino a 63 code per tipo per porta.
 * Falliamo subito con un messaggio chiaro piuttosto che scoprirlo a metà
 * del setup con un errore DOCA criptico.
 */
static bool check_queue_budget(int n_ports)
{
    int rx_per_nic = N_QUEUES_PER_PORT;
    int tx_per_nic = N_QUEUES_PER_PORT * (n_ports - 1);

    printf("Budget code per NIC: RX=%d  TX=%d  (limite HW tipico: 63 per tipo)\n",
           rx_per_nic, tx_per_nic);

    if (rx_per_nic > 63 || tx_per_nic > 63) {
        fprintf(stderr,
            "ERRORE: con n_ports=%d e N_QUEUES_PER_PORT=%d servirebbero %d code TX "
            "per NIC, oltre il limite HW tipico (63).\n"
            "  Riduci N_QUEUES_PER_PORT in gpu_bridge_rss.h oppure il numero di porte (-n).\n",
            n_ports, N_QUEUES_PER_PORT, tx_per_nic);
        return false;
    }
    return true;
}

/* ==========================================================================
 * APERTURA DEVICE NIC (invariata rispetto a v1)
 * ==========================================================================
 */
static doca_error_t open_nic_by_pci(const char *pci_addr, struct doca_dev **dev)
{
    struct doca_devinfo **list;
    uint32_t n, i;
    uint8_t is_equal;
    doca_error_t res;

    res = doca_devinfo_create_list(&list, &n);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "doca_devinfo_create_list: %s\n", doca_error_get_descr(res));
        return res;
    }

    for (i = 0; i < n; i++) {
        res = doca_devinfo_is_equal_pci_addr(list[i], pci_addr, &is_equal);
        if (res == DOCA_SUCCESS && is_equal) {
            res = doca_dev_open(list[i], dev);
            doca_devinfo_destroy_list(list);
            return res;
        }
    }

    doca_devinfo_destroy_list(list);
    fprintf(stderr, "NIC device '%s' non trovato\n"
            "  (ricorda: esegui con 'sudo ip netns exec bf2 ...')\n", pci_addr);
    return DOCA_ERROR_NOT_FOUND;
}

/* ==========================================================================
 * INIT GLOBALE DOCA FLOW (invariata)
 * ==========================================================================
 */
static doca_error_t flow_global_init(void)
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
    if (res != DOCA_SUCCESS)
        fprintf(stderr, "doca_flow_init: %s\n", doca_error_get_descr(res));
    return res;
}

static doca_error_t flow_start_port(struct bridge_port *port, int port_id)
{
    struct doca_flow_port_cfg *port_cfg;
    doca_error_t res;

    res = doca_flow_port_cfg_create(&port_cfg);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_port_cfg_set_port_id(port_cfg, (uint16_t)port_id);
    doca_flow_port_cfg_set_dev(port_cfg, port->ddev);

    res = doca_flow_port_start(port_cfg, &port->flow_port);
    doca_flow_port_cfg_destroy(port_cfg);
    if (res != DOCA_SUCCESS)
        fprintf(stderr, "doca_flow_port_start (port %d): %s\n",
                port_id, doca_error_get_descr(res));
    return res;
}

/* ==========================================================================
 * SETUP DI UNA SINGOLA RXQ (una per coppia porta,coda)
 * ==========================================================================
 * Stessa logica di setup_port_rxq in v1 (buffer ciclico GPU + mmap
 * cross-port registrato su TUTTE le N NIC), ma opera su un bridge_rxq_slot
 * generico invece che direttamente su bridge_port — così può essere
 * richiamata N_QUEUES_PER_PORT volte per porta.
 *
 * owner_ddev: la NIC proprietaria di QUESTA coda (riceve i pacchetti qui).
 * all_ddevs/n_ports: serve per registrare il mmap con TUTTE le NIC, in modo
 *   che qualunque porta di destinazione possa fare DMA READ cross-port.
 */
static doca_error_t setup_rxq_slot(struct bridge_rxq_slot *slot,
                                    struct doca_dev        *owner_ddev,
                                    struct doca_dev       **all_ddevs,
                                    int                     n_ports,
                                    struct doca_gpu        *gdev,
                                    int                     cuda_id,
                                    const char             *label)
{
    doca_error_t res;
    uint32_t cyclic_buf_size = 0;
    size_t page_sz = system_page_size();
    struct cudaDeviceProp prop;

    res = doca_eth_rxq_create(owner_ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &slot->rxq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_create: %s\n", label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_rxq_set_type(slot->rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_type: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    res = doca_eth_rxq_estimate_packet_buf_size(
            DOCA_ETH_RXQ_TYPE_CYCLIC,
            0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, 0, 0,
            &cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] estimate_packet_buf_size: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }
    cyclic_buf_size = (uint32_t)ALIGN_UP(cyclic_buf_size, page_sz);

    res = doca_gpu_mem_alloc(gdev, cyclic_buf_size, page_sz,
                              DOCA_GPU_MEM_TYPE_GPU,
                              &slot->rxq_buf, NULL);
    if (res != DOCA_SUCCESS || !slot->rxq_buf) {
        fprintf(stderr, "[%s] doca_gpu_mem_alloc RXQ: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    res = doca_mmap_create(&slot->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_create: %s\n", label, doca_error_get_descr(res));
        goto err_free_buf_early;
    }

    res = doca_gpu_dmabuf_fd(gdev, slot->rxq_buf, cyclic_buf_size, &slot->rxq_dmabuf_fd);
    if (res != DOCA_SUCCESS) {
        res = doca_mmap_set_memrange(slot->rxq_mmap, slot->rxq_buf, cyclic_buf_size);
    } else {
        res = doca_mmap_set_dmabuf_memrange(slot->rxq_mmap, slot->rxq_dmabuf_fd,
                                             slot->rxq_buf, 0, cyclic_buf_size);
    }
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] mmap set_memrange: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    res = doca_mmap_set_permissions(slot->rxq_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_permissions: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    res = doca_mmap_set_max_num_devices(slot->rxq_mmap, (uint32_t)n_ports);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_max_num_devices: %s\n", label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    for (int q = 0; q < n_ports; q++) {
        res = doca_mmap_add_dev(slot->rxq_mmap, all_ddevs[q]);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_add_dev(porta %d): %s\n",
                    label, q, doca_error_get_descr(res));
            goto err_destroy_mmap;
        }
    }

    res = doca_mmap_start(slot->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_start: %s\n", label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    for (int q = 0; q < n_ports; q++) {
        uint32_t raw_mkey;
        res = doca_mmap_get_mkey(slot->rxq_mmap, all_ddevs[q], &raw_mkey);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_get_mkey(porta %d): %s\n",
                    label, q, doca_error_get_descr(res));
            goto err_free_buf;
        }
        slot->mkey_for_port[q] = htobe32(raw_mkey);
    }

    res = doca_eth_rxq_set_pkt_buf(slot->rxq_cpu, slot->rxq_mmap, 0, cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_pkt_buf: %s\n", label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    cudaGetDeviceProperties(&prop, cuda_id);
    if (prop.major < 9) {
        res = doca_eth_rxq_gpu_enable_mcst_qp(slot->rxq_cpu);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] enable_mcst_qp: %s\n", label, doca_error_get_descr(res));
            goto err_free_buf;
        }
    }

    slot->rxq_ctx = doca_eth_rxq_as_doca_ctx(slot->rxq_cpu);
    if (!slot->rxq_ctx) {
        fprintf(stderr, "[%s] doca_eth_rxq_as_doca_ctx fallito\n", label);
        goto err_free_buf;
    }

    res = doca_ctx_set_datapath_on_gpu(slot->rxq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu RXQ: %s\n", label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    res = doca_ctx_start(slot->rxq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start RXQ: %s\n", label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    res = doca_eth_rxq_get_gpu_handle(slot->rxq_cpu, &slot->rxq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_get_gpu_handle: %s\n", label, doca_error_get_descr(res));
        doca_ctx_stop(slot->rxq_ctx);
        goto err_free_buf;
    }

    return DOCA_SUCCESS;

err_free_buf:
    doca_gpu_mem_free(gdev, slot->rxq_buf);
    slot->rxq_buf = NULL;
    goto err_destroy_rxq;
err_destroy_mmap:
    doca_mmap_destroy(slot->rxq_mmap);
    slot->rxq_mmap = NULL;
err_free_buf_early:
    doca_gpu_mem_free(gdev, slot->rxq_buf);
    slot->rxq_buf = NULL;
err_destroy_rxq:
    doca_eth_rxq_destroy(slot->rxq_cpu);
    slot->rxq_cpu = NULL;
    return DOCA_ERROR_BAD_STATE;
}

/* ==========================================================================
 * SETUP DI UNA SINGOLA TXQ (privata: una per porta_sorgente,coda,porta_dst)
 * ==========================================================================
 * La TXQ non ha buffer dati propri: i WQE del kernel puntano al buffer
 * rxq della sorgente (zero-copy cross-port). Va creata sulla NIC della
 * porta di DESTINAZIONE (tx_ddev), con un queue_id univoco su quella NIC.
 */
static doca_error_t setup_txq_slot(struct bridge_txq_slot *slot,
                                    struct doca_dev        *tx_ddev,
                                    struct doca_gpu         *gdev,
                                    uint16_t                 queue_id,
                                    const char               *label)
{
    doca_error_t res;

    res = doca_eth_txq_create(tx_ddev, MAX_SQ_DESCR_NUM, &slot->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_create: %s\n", label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_txq_set_l3_chksum_offload(slot->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l3_chksum_offload: %s\n", label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_set_l4_chksum_offload(slot->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l4_chksum_offload: %s\n", label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_gpu_set_completion_on_gpu(slot->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_completion_on_gpu: %s\n", label, doca_error_get_descr(res));
        goto err;
    }

    slot->txq_ctx = doca_eth_txq_as_doca_ctx(slot->txq_cpu);
    if (!slot->txq_ctx) {
        fprintf(stderr, "[%s] doca_eth_txq_as_doca_ctx fallito\n", label);
        goto err;
    }

    res = doca_ctx_set_datapath_on_gpu(slot->txq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu TXQ: %s\n", label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_ctx_start(slot->txq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start TXQ: %s\n", label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_apply_queue_id(slot->txq_cpu, queue_id);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] txq_apply_queue_id: %s\n", label, doca_error_get_descr(res));
        doca_ctx_stop(slot->txq_ctx);
        goto err;
    }

    res = doca_eth_txq_get_gpu_handle(slot->txq_cpu, &slot->txq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_get_gpu_handle: %s\n", label, doca_error_get_descr(res));
        doca_ctx_stop(slot->txq_ctx);
        goto err;
    }

    return DOCA_SUCCESS;

err:
    doca_eth_txq_destroy(slot->txq_cpu);
    slot->txq_cpu = NULL;
    return DOCA_ERROR_BAD_STATE;
}

/* ==========================================================================
 * SETUP DOCA FLOW — BASIC ROOT PIPE WILDCARD → RSS SU N_QUEUES_PER_PORT CODE
 * ==========================================================================
 * Va chiamata DOPO che tutte le rxq[q] della porta sono state create
 * (usa rxq[q].rxq_cpu per applicare il queue_id RSS-visibile).
 */
static doca_error_t setup_port_flow(struct bridge_port *port, const char *label)
{
    struct doca_flow_match    match    = {0};
    struct doca_flow_fwd      fwd      = {0};
    struct doca_flow_fwd      miss_fwd = {0};
    struct doca_flow_monitor  mon      = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED
    };
    struct doca_flow_pipe_cfg *pipe_cfg;
    uint16_t rss_queues[N_QUEUES_PER_PORT];
    doca_error_t res;

    for (int q = 0; q < N_QUEUES_PER_PORT; q++) {
        res = doca_eth_rxq_apply_queue_id(port->rxq[q].rxq_cpu, (uint16_t)q);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_eth_rxq_apply_queue_id(coda %d): %s\n",
                    label, q, doca_error_get_descr(res));
            return res;
        }
        rss_queues[q] = (uint16_t)q;
    }

    /* RSS su tutte le N_QUEUES_PER_PORT code, hash su IPv4/IPv6/UDP/TCP.
     * Traffico non-IP (ARP, ecc.) non ha campi su cui la NIC possa fare
     * hash: finisce su una coda fissa (vedi RSS_HASH_FIELDS in header). */
    fwd.type             = DOCA_FLOW_FWD_RSS;
    fwd.rss_type         = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array = rss_queues;
    fwd.rss.nr_queues    = N_QUEUES_PER_PORT;
    fwd.rss.outer_flags  = RSS_HASH_FIELDS;

    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, port->flow_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,    "BRIDGE_L2_PIPE_RSS");
    doca_flow_pipe_cfg_set_type(pipe_cfg,    DOCA_FLOW_PIPE_BASIC);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
    doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &port->root_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_create: %s\n", label, doca_error_get_descr(res));
        return res;
    }

    res = doca_flow_pipe_basic_add_entry(
        0, port->root_pipe, &match, 0,
        NULL, NULL, NULL,
        DOCA_FLOW_ENTRY_FLAGS_NO_WAIT,
        NULL, &port->root_entry);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_basic_add_entry: %s\n", label, doca_error_get_descr(res));
        return res;
    }

    res = doca_flow_entries_process(port->flow_port, 0, 0, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_entries_process: %s\n", label, doca_error_get_descr(res));
        return res;
    }

    printf("[%s] DOCA Flow pronto — BASIC ROOT wildcard → RSS su %d code\n",
           label, N_QUEUES_PER_PORT);
    return DOCA_SUCCESS;
}

/* ==========================================================================
 * CLEANUP
 * ==========================================================================
 * txq è un array [MAX_N_PORTS][N_QUEUES_PER_PORT][MAX_N_PORTS] indicizzato
 * [porta_sorgente][coda][porta_dst]; txq[p][q][p] non è mai stato inizializzato
 * (si salta sempre dst==src, vedi main) quindi va ignorato in cleanup.
 */
static void cleanup_all(struct bridge_port      *ports,
                         int                      n_ports,
                         struct bridge_txq_slot (*txq)[N_QUEUES_PER_PORT][MAX_N_PORTS],
                         struct doca_gpu         *gdev,
                         struct bridge_queue_ctx *gpu_queues,
                         uint64_t                *mac_table_gpu,
                         uint32_t                *gpu_exit,
                         uint64_t                *gpu_fwd,
                         uint64_t                *gpu_rx_total,
                         uint64_t                *gpu_flood,
                         uint64_t                *gpu_unicast,
                         uint64_t                *gpu_drop,
                         struct mac_flap_record  *gpu_flap_ring,
                         uint32_t                *gpu_flap_head,
                         cudaStream_t             stream,
                         bool                     flow_initialized)
{
    if (stream)        cudaStreamDestroy(stream);
    if (mac_table_gpu) cudaFree(mac_table_gpu);
    if (gpu_queues)    cudaFree(gpu_queues);

    if (gpu_exit && gdev) doca_gpu_mem_free(gdev, gpu_exit);
    if (gpu_fwd  && gdev) doca_gpu_mem_free(gdev, gpu_fwd);

    if (gpu_rx_total && gdev) doca_gpu_mem_free(gdev, gpu_rx_total);
    if (gpu_flood    && gdev) doca_gpu_mem_free(gdev, gpu_flood);
    if (gpu_unicast  && gdev) doca_gpu_mem_free(gdev, gpu_unicast);
    if (gpu_drop     && gdev) doca_gpu_mem_free(gdev, gpu_drop);
    if (gpu_flap_ring && gdev) doca_gpu_mem_free(gdev, gpu_flap_ring);
    if (gpu_flap_head && gdev) doca_gpu_mem_free(gdev, gpu_flap_head);

    for (int p = 0; p < n_ports; p++) {
        if (ports[p].root_pipe) doca_flow_pipe_destroy(ports[p].root_pipe);
        if (ports[p].flow_port) doca_flow_port_stop(ports[p].flow_port);

        for (int q = 0; q < N_QUEUES_PER_PORT; q++) {
            for (int d = 0; d < n_ports; d++) {
                if (d == p) continue;
                struct bridge_txq_slot *t = &txq[p][q][d];
                if (t->txq_ctx) doca_ctx_stop(t->txq_ctx);
                if (t->txq_cpu) doca_eth_txq_destroy(t->txq_cpu);
            }

            struct bridge_rxq_slot *r = &ports[p].rxq[q];
            if (r->rxq_ctx)  doca_ctx_stop(r->rxq_ctx);
            if (r->rxq_buf && gdev) doca_gpu_mem_free(gdev, r->rxq_buf);
            if (r->rxq_cpu)  doca_eth_rxq_destroy(r->rxq_cpu);
            if (r->rxq_mmap) doca_mmap_destroy(r->rxq_mmap);
        }

        if (ports[p].ddev) doca_dev_close(ports[p].ddev);
    }

    if (gdev)             doca_gpu_destroy(gdev);
    if (flow_initialized) doca_flow_destroy();
}

/* ==========================================================================
 * USAGE
 * ==========================================================================
 */
static void usage(const char *prog)
{
    fprintf(stderr,
        "Uso: %s -n <pci_porta0> -n <pci_porta1> [-n <pci_portaN>...] "
        "-g <pci_gpu> [-i <cuda_id>]\n"
        "\n"
        "  -n  PCI address di una porta NIC (ripetere per ogni porta, min 2, max %d)\n"
        "  -g  PCI address della GPU\n"
        "  -i  CUDA device index (default: 0)\n"
        "\n"
        "Code RSS per porta: %d (fisso a compile time, vedi N_QUEUES_PER_PORT\n"
        "in gpu_bridge_rss.h). Un blocco CUDA dedicato per ogni coppia (porta, coda).\n"
        "\n"
        "Esempio (BF2, 2 porte):\n"
        "  sudo ip netns exec bf2 ./%s -n ad:00.0 -n ad:00.1 -g b0:00.0\n",
        prog, MAX_N_PORTS, N_QUEUES_PER_PORT, prog);
}

/* ==========================================================================
 * MAIN
 * ==========================================================================
 */
int main(int argc, char *argv[])
{
    char nic_pci[MAX_N_PORTS][DOCA_DEVINFO_PCI_ADDR_SIZE];
    char gpu_pci[64] = {0};
    int  cuda_id    = 0;
    int  n_ports    = 0;
    int  opt;

    memset(nic_pci, 0, sizeof(nic_pci));

    while ((opt = getopt(argc, argv, "n:g:i:h")) != -1) {
        switch (opt) {
        case 'n':
            if (n_ports >= MAX_N_PORTS) {
                fprintf(stderr, "Errore: max %d flag -n\n", MAX_N_PORTS);
                return 1;
            }
            strncpy(nic_pci[n_ports++], optarg, DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
            break;
        case 'g':
            strncpy(gpu_pci, optarg, sizeof(gpu_pci) - 1);
            break;
        case 'i':
            cuda_id = atoi(optarg);
            break;
        case 'h': default:
            usage(argv[0]); return 1;
        }
    }

    if (n_ports < 2 || !gpu_pci[0]) {
        fprintf(stderr, "Errore: servono almeno 2 flag -n e 1 flag -g.\n");
        usage(argv[0]);
        return 1;
    }

    if (!check_queue_budget(n_ports)) {
        usage(argv[0]);
        return 1;
    }

    /* ── Variabili principali ─────────────────────────────────────────── */
    doca_error_t          res          = DOCA_SUCCESS;
    static struct bridge_port ports[MAX_N_PORTS];
    struct doca_dev      *all_ddevs[MAX_N_PORTS];
    struct doca_gpu      *gdev         = NULL;

    /* TXQ private [porta_sorgente][coda][porta_dst]. Statico: troppo
     * grande per lo stack (MAX_N_PORTS^2 * N_QUEUES_PER_PORT strutture). */
    static struct bridge_txq_slot txq[MAX_N_PORTS][N_QUEUES_PER_PORT][MAX_N_PORTS];

    /* Array host dei contesti-coda, copiato in GPU per il kernel. */
    static struct bridge_queue_ctx queues_host[MAX_N_PORTS * N_QUEUES_PER_PORT];
    struct bridge_queue_ctx *gpu_queues = NULL;

    uint64_t             *mac_table_gpu = NULL;
    uint32_t             *gpu_exit     = NULL;
    uint32_t             *cpu_exit     = NULL;
    uint64_t             *gpu_fwd      = NULL;
    uint64_t             *cpu_fwd      = NULL;
    uint64_t             *gpu_rx_total = NULL;
    uint64_t             *cpu_rx_total = NULL;
    uint64_t             *gpu_flood    = NULL;
    uint64_t             *cpu_flood    = NULL;
    uint64_t             *gpu_unicast  = NULL;
    uint64_t             *cpu_unicast  = NULL;
    uint64_t             *gpu_drop     = NULL;
    uint64_t             *cpu_drop     = NULL;
    struct mac_flap_record *gpu_flap_ring = NULL;
    struct mac_flap_record *cpu_flap_ring = NULL;
    uint32_t             *gpu_flap_head = NULL;
    uint32_t             *cpu_flap_head = NULL;
    cudaStream_t          stream       = NULL;
    bool                  flow_inited  = false;
    struct bridge_kernel_params kp     = {0};

    memset(ports,    0, sizeof(ports));
    memset(all_ddevs, 0, sizeof(all_ddevs));
    memset(txq, 0, sizeof(txq));
    memset(queues_host, 0, sizeof(queues_host));

    cudaSetDevice(cuda_id);

    /* ── 1. Apri TUTTE le NIC ─────────────────────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        res = open_nic_by_pci(nic_pci[p], &ports[p].ddev);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "Apertura NIC %s fallita\n", nic_pci[p]);
            for (int j = 0; j < p; j++) doca_dev_close(ports[j].ddev);
            return 1;
        }
        all_ddevs[p] = ports[p].ddev;
        printf("NIC porta %d aperta: %s\n", p, nic_pci[p]);
    }

    /* ── 2. Init globale DOCA Flow ────────────────────────────────────── */
    res = flow_global_init();
    if (res != DOCA_SUCCESS) goto cleanup;
    flow_inited = true;

    /* ── 3. Flow port per ogni porta ─────────────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        res = flow_start_port(&ports[p], p);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 4. Crea GPU context ─────────────────────────────────────────── */
    res = doca_gpu_create(gpu_pci, &gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "doca_gpu_create (%s): %s\n", gpu_pci, doca_error_get_descr(res));
        goto cleanup;
    }
    printf("GPU %s (CUDA device %d) aperta\n", gpu_pci, cuda_id);

    /* ── 5. N_QUEUES_PER_PORT RXQ per ogni porta ─────────────────────────
     * Ogni rxq viene registrata con TUTTI gli N ddev (cross-port). */
    char label[48];
    for (int p = 0; p < n_ports; p++) {
        for (int q = 0; q < N_QUEUES_PER_PORT; q++) {
            snprintf(label, sizeof(label), "porta %d coda %d", p, q);
            res = setup_rxq_slot(&ports[p].rxq[q], ports[p].ddev,
                                  all_ddevs, n_ports, gdev, cuda_id, label);
            if (res != DOCA_SUCCESS) goto cleanup;
        }
    }

    /* ── 6. TXQ private per ogni (porta_sorgente, coda, porta_dst) ────────
     * Creata sulla NIC di DESTINAZIONE, con queue_id univoco su quella NIC
     * (ports[d].next_txq_id, incrementato ogni volta). */
    for (int p = 0; p < n_ports; p++) {
        for (int q = 0; q < N_QUEUES_PER_PORT; q++) {
            for (int d = 0; d < n_ports; d++) {
                if (d == p) continue;
                snprintf(label, sizeof(label), "porta %d coda %d -> porta %d", p, q, d);
                res = setup_txq_slot(&txq[p][q][d], ports[d].ddev, gdev,
                                      ports[d].next_txq_id++, label);
                if (res != DOCA_SUCCESS) goto cleanup;
            }
        }
    }

    /* ── 7. DOCA Flow pipe RSS per ogni porta ──────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        snprintf(label, sizeof(label), "porta %d", p);
        res = setup_port_flow(&ports[p], label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 8. Alloca MAC table (condivisa da tutti i blocchi) ─────────────── */
    if (cudaMalloc((void **)&mac_table_gpu,
                   (size_t)MAC_TABLE_SIZE * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc MAC table fallita\n");
        goto cleanup;
    }
    cudaMemset(mac_table_gpu, 0, (size_t)MAC_TABLE_SIZE * sizeof(uint64_t));

    /* ── 9. exit_cond (GPU_CPU) ──────────────────────────────────────── */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit, (void **)&cpu_exit);
    if (res != DOCA_SUCCESS || !gpu_exit) {
        fprintf(stderr, "alloc exit_cond: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_exit = 0;

    /* ── 10. fwd_count e contatori diagnostici (CPU_GPU) ─────────────────
     * Ora aggregati con atomicAdd da TUTTI i blocchi (vedi kernel). */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_fwd, (void **)&cpu_fwd);
    if (res != DOCA_SUCCESS || !gpu_fwd) {
        fprintf(stderr, "alloc fwd_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_fwd = 0;

    res = doca_gpu_mem_alloc(gdev, (size_t)n_ports * sizeof(uint64_t),
                              system_page_size(), DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_rx_total, (void **)&cpu_rx_total);
    if (res != DOCA_SUCCESS || !gpu_rx_total) {
        fprintf(stderr, "alloc rx_pkt_total: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    memset(cpu_rx_total, 0, (size_t)n_ports * sizeof(uint64_t));

    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_flood, (void **)&cpu_flood);
    if (res != DOCA_SUCCESS || !gpu_flood) {
        fprintf(stderr, "alloc flood_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_flood = 0;

    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_unicast, (void **)&cpu_unicast);
    if (res != DOCA_SUCCESS || !gpu_unicast) {
        fprintf(stderr, "alloc unicast_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_unicast = 0;

    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_drop, (void **)&cpu_drop);
    if (res != DOCA_SUCCESS || !gpu_drop) {
        fprintf(stderr, "alloc drop_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_drop = 0;

    res = doca_gpu_mem_alloc(gdev, (size_t)FLAP_RING_SIZE * sizeof(struct mac_flap_record),
                              system_page_size(), DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_flap_ring, (void **)&cpu_flap_ring);
    if (res != DOCA_SUCCESS || !gpu_flap_ring) {
        fprintf(stderr, "alloc flap_ring: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    memset(cpu_flap_ring, 0, (size_t)FLAP_RING_SIZE * sizeof(struct mac_flap_record));

    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_flap_head, (void **)&cpu_flap_head);
    if (res != DOCA_SUCCESS || !gpu_flap_head) {
        fprintf(stderr, "alloc flap_ring_head: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_flap_head = 0;

    /* ── 11. CUDA stream ─────────────────────────────────────────────── */
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags fallita\n");
        goto cleanup;
    }

    /* ── 12. Costruisci l'array queues_host e copialo in GPU ─────────────
     * Un elemento per ogni blocco CUDA, indicizzato come p*N_QUEUES_PER_PORT+q
     * (deve combaciare con l'ordine di lancio <<<n_blocks,...>>> nel kernel,
     * dove blockIdx.x indicizza direttamente questo array). */
    for (int p = 0; p < n_ports; p++) {
        for (int q = 0; q < N_QUEUES_PER_PORT; q++) {
            struct bridge_queue_ctx *c = &queues_host[p * N_QUEUES_PER_PORT + q];
            c->port     = p;
            c->queue    = q;
            c->rxq_gpu  = ports[p].rxq[q].rxq_gpu;
            for (int d = 0; d < n_ports; d++) {
                if (d == p) {
                    c->txq_gpu[d] = NULL;
                    c->rxq_mkey_for_dst[d] = 0;
                    continue;
                }
                c->txq_gpu[d]           = txq[p][q][d].txq_gpu;
                c->rxq_mkey_for_dst[d]  = ports[p].rxq[q].mkey_for_port[d];
            }
        }
    }

    if (cudaMalloc((void **)&gpu_queues,
                   (size_t)(n_ports * N_QUEUES_PER_PORT) * sizeof(struct bridge_queue_ctx)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc queues_host fallita\n");
        goto cleanup;
    }
    if (cudaMemcpy(gpu_queues, queues_host,
                   (size_t)(n_ports * N_QUEUES_PER_PORT) * sizeof(struct bridge_queue_ctx),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy queues_host fallita\n");
        goto cleanup;
    }

    /* ── 13. Popola bridge_kernel_params ─────────────────────────────── */
    kp.n_ports   = n_ports;
    kp.n_queues  = N_QUEUES_PER_PORT;
    kp.queues    = gpu_queues;
    kp.mac_table = mac_table_gpu;
    kp.exit_cond = gpu_exit;
    kp.fwd_count = gpu_fwd;

    kp.rx_pkt_total   = gpu_rx_total;
    kp.flood_count    = gpu_flood;
    kp.unicast_count  = gpu_unicast;
    kp.drop_count     = gpu_drop;
    kp.flap_ring      = gpu_flap_ring;
    kp.flap_ring_head = gpu_flap_head;

    /* ── 14. Lancia il kernel CUDA persistente ────────────────────────── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\nGPU Bridge RSS a %d porte, %d code/porta (%d blocchi CUDA) avviato:\n",
           n_ports, N_QUEUES_PER_PORT, n_ports * N_QUEUES_PER_PORT);
    for (int p = 0; p < n_ports; p++)
        printf("  porta %d: %s\n", p, nic_pci[p]);
    printf("  GPU:     %s (CUDA device %d)\n", gpu_pci, cuda_id);
    printf("  MAC table: %d slot (FNV-1a, linear probing, 8-bit port, condivisa da tutti i blocchi)\n",
           MAC_TABLE_SIZE);
    printf("Premi Ctrl+C per fermare.\n\n");

    res = kernel_launch_bridge(stream, &kp);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "kernel_launch_bridge: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }

    /* ── 15. Aspetta Ctrl+C, stampando statistiche live ─────────────────
     * Stessa logica di v1: i contatori sono ora aggregati con atomicAdd
     * da tutti i blocchi, ma restano in memoria CPU_GPU coerente grazie
     * a __threadfence_system() lato GPU — nessuna sync/memcpy richiesta qui. */
    {
        uint32_t last_flap_head = 0;
        while (!DOCA_GPUNETIO_VOLATILE(g_force_quit)) {
            usleep(500000);

            uint64_t rx_sum = 0;
            for (int p = 0; p < n_ports; p++)
                rx_sum += cpu_rx_total[p];

            printf("[live] rx_tot=%lu (", rx_sum);
            for (int p = 0; p < n_ports; p++)
                printf("porta%d=%lu%s", p, cpu_rx_total[p], p + 1 < n_ports ? " " : "");
            printf(")  fwd=%lu  flood=%lu  unicast=%lu  drop=%lu  mac_flaps=%u\n",
                   *cpu_fwd, *cpu_flood, *cpu_unicast, *cpu_drop, *cpu_flap_head);

            uint32_t head = *cpu_flap_head;
            uint32_t new_flaps = head - last_flap_head;
            if (new_flaps > 0) {
                uint32_t show = new_flaps > FLAP_RING_SIZE ? FLAP_RING_SIZE : new_flaps;
                if (new_flaps > FLAP_RING_SIZE)
                    printf("  (%u flap non mostrati, ring overflow)\n",
                           new_flaps - FLAP_RING_SIZE);
                for (uint32_t i = 0; i < show; i++) {
                    uint32_t seq_want = head - show + i;
                    struct mac_flap_record *r = &cpu_flap_ring[seq_want % FLAP_RING_SIZE];
                    printf("  FLAP mac=%s  porta %u -> porta %u  (seq %lu)\n",
                           mac48_to_str(r->mac48), r->old_port, r->new_port,
                           (unsigned long)r->seq);
                }
                last_flap_head = head;
            }

            fflush(stdout);
        }
    }

    /* ── 16. Ferma il kernel ────────────────────────────────────────── */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit) = 1;
    cudaStreamSynchronize(stream);

    printf("\nFermato. Totale pacchetti forwardati: %lu\n", *cpu_fwd);
    printf("  Ricevuti per porta: ");
    for (int p = 0; p < n_ports; p++)
        printf("porta%d=%lu ", p, cpu_rx_total[p]);
    printf("\n  flood=%lu  unicast=%lu  drop=%lu  mac_flaps totali=%u\n",
           *cpu_flood, *cpu_unicast, *cpu_drop, *cpu_flap_head);

cleanup:
    cleanup_all(ports, n_ports, txq, gdev, gpu_queues, mac_table_gpu,
                gpu_exit, gpu_fwd,
                gpu_rx_total, gpu_flood, gpu_unicast, gpu_drop,
                gpu_flap_ring, gpu_flap_head,
                stream, flow_inited);
    printf("Done.\n");
    return (res == DOCA_SUCCESS) ? 0 : 1;
}
