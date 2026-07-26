/*
 * gpu_bridge.c — setup lato CPU per il GPU L2 Bridge a N porte.
 *
 * Questo file gestisce tutto ciò che riguarda il CPU:
 *   - apertura di tutti i device DOCA (NIC e GPU)
 *   - allocazione delle code di ricezione (RXQ) e trasmissione (TXQ)
 *   - configurazione di DOCA Flow per lo steering dei pacchetti
 *   - allocazione della MAC table e delle variabili di sincronizzazione
 *   - lancio del kernel CUDA persistente
 *   - attesa di Ctrl+C e cleanup
 *
 * Il kernel CUDA (gpu_bridge_kernel.cu) gestisce il datapath real-time:
 * ricezione, MAC learning, FIB lookup, forward/drop/flood.
 *
 * Utilizzo:
 *   sudo ip netns exec bf2 ./gpu_bridge \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0
 *
 * Note:
 *   - Eseguire nel netns bf2: DOCA usa libibverbs (namespace-aware)
 *   - Il traffico test arriva sempre dall'Intel 810 verso la BF2
 *   - MAX_N_PORTS porte fisiche supportate (compile-time), n_ports a runtime
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

#include "gpu_bridge.h"

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

/* Formatta un mac48 (bit 0-47, byte 0 nei bit alti — vedi mac_to_u64 nel
 * kernel) come stringa "aa:bb:cc:dd:ee:ff" in un buffer statico. */
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
 * APERTURA DEVICE NIC
 * ==========================================================================
 * Scansiona tutti i DOCA device e apre quello con l'indirizzo PCI dato.
 * doca_devinfo_is_equal_pci_addr gestisce "ad:00.0" e "0000:ad:00.0".
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
 * INIT GLOBALE DOCA FLOW
 * ==========================================================================
 * Chiamato UNA SOLA VOLTA. "vnf" = Virtual Network Function.
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

/* ==========================================================================
 * AVVIO PORTA DOCA FLOW
 * ==========================================================================
 * Ogni porta fisica ha un port_id univoco (0..n_ports-1).
 */
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
 * SETUP CODA DI RICEZIONE (RXQ)
 * ==========================================================================
 * Crea e avvia la RXQ GPU per la porta port_idx.
 *
 * CROSS-PORT MMAP A N PORTE:
 *   Il buffer GPU di questa porta deve essere leggibile dalla NIC di OGNI
 *   altra porta (per zero-copy cross-port forwarding e flooding).
 *
 *   Registriamo il mmap con TUTTI gli N ddev (incluso il proprio):
 *     doca_mmap_add_dev(mmap, all_ddevs[0])
 *     doca_mmap_add_dev(mmap, all_dde
 * vs[1])
 *     ...
 *     doca_mmap_add_dev(mmap, all_ddevs[N-1])
 *   poi doca_mmap_start() registra con tutti i PD RDMA contemporaneamente.
 *
 *   Dopo start, otteniamo un mkey separato per ogni NIC:
 *     doca_mmap_get_mkey(mmap, all_ddevs[q], &mkey) → port->rxq_mkey_for_port[q]
 *
 *   rxq_mkey_for_port[q] = mkey che autorizza la NIC porta q a fare DMA READ
 *   sul buffer GPU di questa porta (nei WQE di txq[q]).
 *
 * Parametri:
 *   port       — porta da configurare
 *   all_ddevs  — array di tutti i ddev[0..n_ports-1]
 *   n_ports    — numero totale di porte
 *   gdev       — handle GPU DOCA
 *   cuda_id    — CUDA device index
 *   port_label — stringa per i log
 */
static doca_error_t setup_port_rxq(struct bridge_port  *port,
                                    struct doca_dev    **all_ddevs,
                                    int                  n_ports,
                                    struct doca_gpu     *gdev,
                                    int                  cuda_id,
                                    const char          *port_label)
{
    doca_error_t res;
    uint32_t cyclic_buf_size = 0;
    size_t page_sz = system_page_size();
    struct cudaDeviceProp prop;

    /* ── Crea RXQ CYCLIC ─────────────────────────────────────────────── */
    res = doca_eth_rxq_create(port->ddev, MAX_PKT_NUM, MAX_PKT_SIZE,
                               &port->rxq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_create: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_rxq_set_type(port->rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_type: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    /* ── Calcola dimensione buffer (DOCA considera header interni e padding) */
    res = doca_eth_rxq_estimate_packet_buf_size(
            DOCA_ETH_RXQ_TYPE_CYCLIC,
            0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, 0, 0,
            &cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] estimate_packet_buf_size: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }
    cyclic_buf_size = (uint32_t)ALIGN_UP(cyclic_buf_size, page_sz);

    /* ── Alloca buffer GPU (VRAM A30X) ───────────────────────────────────
     * DOCA_GPU_MEM_TYPE_GPU: memoria esclusivamente GPU.
     * La NIC accede via DMA peer-to-peer (PCIe), non tramite CPU.
     * Deve essere allocato PRIMA di doca_mmap_add_dev: DOCA richiede che
     * il memrange sia configurato prima di poter aggiungere device al mmap.
     */
    res = doca_gpu_mem_alloc(gdev, cyclic_buf_size, page_sz,
                              DOCA_GPU_MEM_TYPE_GPU,
                              &port->rxq_buf, NULL);
    if (res != DOCA_SUCCESS || !port->rxq_buf) {
        fprintf(stderr, "[%s] doca_gpu_mem_alloc RXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    /* ── Crea mmap ────────────────────────────────────────────────────── */
    res = doca_mmap_create(&port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_create: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf_early;
    }

    /* ── Configura memrange PRIMA di add_dev ─────────────────────────────
     * DMABuf: kernel >= 5.12, nessun modulo aggiuntivo richiesto.
     * nvidia-peermem: metodo tradizionale, richiede gdrcopy/gdrdrv.
     */
    res = doca_gpu_dmabuf_fd(gdev, port->rxq_buf, cyclic_buf_size,
                              &port->rxq_dmabuf_fd);
    if (res != DOCA_SUCCESS) {
        printf("[%s] DMABuf non disponibile, uso nvidia-peermem\n", port_label);
        res = doca_mmap_set_memrange(port->rxq_mmap,
                                      port->rxq_buf, cyclic_buf_size);
    } else {
        res = doca_mmap_set_dmabuf_memrange(port->rxq_mmap,
                                             port->rxq_dmabuf_fd,
                                             port->rxq_buf, 0, cyclic_buf_size);
    }
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] mmap set_memrange: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    res = doca_mmap_set_permissions(port->rxq_mmap,
                                     DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_permissions: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* ── Imposta max device PRIMA di add_dev ─────────────────────────────
     * Per default DOCA alloca spazio per 1 solo device.
     * Senza questa chiamata il secondo add_dev fallisce con "Memory allocation
     * failure" perché la struttura interna non ha spazio per più entry.
     * (Pattern dal campione ufficiale NVIDIA eth_l2_fwd.)
     */
    res = doca_mmap_set_max_num_devices(port->rxq_mmap, (uint32_t)n_ports);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_max_num_devices: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* ── Registra con TUTTI gli N ddev (cross-port generico) ────────────
     * Ogni NIC ottiene il proprio mkey per fare DMA READ sul buffer GPU.
     */
    for (int q = 0; q < n_ports; q++) {
        res = doca_mmap_add_dev(port->rxq_mmap, all_ddevs[q]);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_add_dev(porta %d): %s\n",
                    port_label, q, doca_error_get_descr(res));
            goto err_destroy_mmap;
        }
    }

    /* ── Start mmap: registra il buffer in TUTTI i PD RDMA aggiunti ─────
     * Da questo momento ogni NIC ha una chiave hardware per il DMA.
     */
    res = doca_mmap_start(port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_start: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Ottieni mkey separato per ogni NIC ────────────────────────────
     * rxq_mkey_for_port[q] = mkey che autorizza la NIC porta q a fare DMA
     * READ sul buffer GPU di questa porta (zero-copy cross-port generico).
     * htobe32: formato big-endian richiesto dal WQE InfiniBand/RDMA.
     */
    for (int q = 0; q < n_ports; q++) {
        uint32_t raw_mkey;
        res = doca_mmap_get_mkey(port->rxq_mmap, all_ddevs[q], &raw_mkey);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_get_mkey(porta %d): %s\n",
                    port_label, q, doca_error_get_descr(res));
            goto err_free_buf;
        }
        port->rxq_mkey_for_port[q] = htobe32(raw_mkey);
    }
    printf("[%s] mkey[0]=0x%08x mkey[1]=0x%08x\n",
           port_label, port->rxq_mkey_for_port[0],
           n_ports > 1 ? port->rxq_mkey_for_port[1] : 0);

    /* ── Collega buffer alla RXQ ─────────────────────────────────────── */
    res = doca_eth_rxq_set_pkt_buf(port->rxq_cpu, port->rxq_mmap, 0, cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_pkt_buf: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Pre-Hopper GPU (A30X = sm_80): abilita multicast QP ────────────
     * GPU con compute capability < 9 richiedono multicast QP workaround.
     */
    cudaGetDeviceProperties(&prop, cuda_id);
    if (prop.major < 9) {
        res = doca_eth_rxq_gpu_enable_mcst_qp(port->rxq_cpu);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] enable_mcst_qp: %s\n",
                    port_label, doca_error_get_descr(res));
            goto err_free_buf;
        }
    }

    /* ── Imposta GPU come destinazione del datapath ──────────────────── */
    port->rxq_ctx = doca_eth_rxq_as_doca_ctx(port->rxq_cpu);
    if (!port->rxq_ctx) {
        fprintf(stderr, "[%s] doca_eth_rxq_as_doca_ctx fallito\n", port_label);
        goto err_free_buf;
    }

    res = doca_ctx_set_datapath_on_gpu(port->rxq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu RXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Avvia il contesto: RXQ ora hardware-active ─────────────────── */
    res = doca_ctx_start(port->rxq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start RXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Handle GPU per il kernel CUDA ─────────────────────────────── */
    res = doca_eth_rxq_get_gpu_handle(port->rxq_cpu, &port->rxq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_get_gpu_handle: %s\n",
                port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->rxq_ctx);
        goto err_free_buf;
    }

    printf("[%s] RXQ pronta — buf %p  size %u B\n",
           port_label, port->rxq_buf, cyclic_buf_size);
    return DOCA_SUCCESS;

err_free_buf:
    doca_gpu_mem_free(gdev, port->rxq_buf);
    port->rxq_buf = NULL;
    goto err_destroy_rxq;
err_destroy_mmap:
    doca_mmap_destroy(port->rxq_mmap);
    port->rxq_mmap = NULL;
err_free_buf_early:
    doca_gpu_mem_free(gdev, port->rxq_buf);
    port->rxq_buf = NULL;
err_destroy_rxq:
    doca_eth_rxq_destroy(port->rxq_cpu);
    port->rxq_cpu = NULL;
    return DOCA_ERROR_BAD_STATE;
}

/* ==========================================================================
 * SETUP CODA DI TRASMISSIONE (TXQ)
 * ==========================================================================
 * La TXQ NON ha buffer dati propri: i WQE del kernel CUDA puntano
 * al buffer rxq di un'altra porta (zero-copy cross-port).
 */
static doca_error_t setup_port_txq(struct bridge_port *port,
                                    struct doca_gpu    *gdev,
                                    const char         *port_label)
{
    doca_error_t res;

    res = doca_eth_txq_create(port->ddev, MAX_SQ_DESCR_NUM, &port->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_create: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_txq_set_l3_chksum_offload(port->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l3_chksum_offload: %s\n",
                port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_set_l4_chksum_offload(port->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l4_chksum_offload: %s\n",
                port_label, doca_error_get_descr(res));
        goto err;
    }

    /* CQE in GPU memory: il kernel fa polling senza coinvolgere la CPU */
    res = doca_eth_txq_gpu_set_completion_on_gpu(port->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_completion_on_gpu: %s\n",
                port_label, doca_error_get_descr(res));
        goto err;
    }

    port->txq_ctx = doca_eth_txq_as_doca_ctx(port->txq_cpu);
    if (!port->txq_ctx) {
        fprintf(stderr, "[%s] doca_eth_txq_as_doca_ctx fallito\n", port_label);
        goto err;
    }

    res = doca_ctx_set_datapath_on_gpu(port->txq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu TXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_ctx_start(port->txq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start TXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_apply_queue_id(port->txq_cpu, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] txq_apply_queue_id: %s\n",
                port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->txq_ctx);
        goto err;
    }

    res = doca_eth_txq_get_gpu_handle(port->txq_cpu, &port->txq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_get_gpu_handle: %s\n",
                port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->txq_ctx);
        goto err;
    }

    printf("[%s] TXQ pronta — %u WQE\n", port_label, MAX_SQ_DESCR_NUM);
    return DOCA_SUCCESS;

err:
    doca_eth_txq_destroy(port->txq_cpu);
    port->txq_cpu = NULL;
    return DOCA_ERROR_BAD_STATE;
}

/* ==========================================================================
 * SETUP DOCA FLOW — BASIC ROOT PIPE WILDCARD
 * ==========================================================================
 * Una BASIC ROOT pipe con match={0} (wildcard completo) cattura TUTTO
 * il traffico L2 (ARP, IPv4, IPv6, VLAN, ...) e lo invia alla GPU RXQ.
 *
 * Chiamata DOPO setup_port_rxq (usa port->rxq_cpu) e flow_start_port.
 */
static doca_error_t setup_port_flow(struct bridge_port *port,
                                     const char         *port_label)
{
    struct doca_flow_match    match    = {0};
    struct doca_flow_fwd      fwd      = {0};
    struct doca_flow_fwd      miss_fwd = {0};
    struct doca_flow_monitor  mon      = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED
    };
    struct doca_flow_pipe_cfg *pipe_cfg;
    uint16_t rss_queue[1] = {0};
    doca_error_t res;

    /* Associa RXQ alla queue 0 di DOCA Flow (deve essere prima di pipe_create) */
    res = doca_eth_rxq_apply_queue_id(port->rxq_cpu, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_apply_queue_id: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    /* RSS verso queue 0 (la GPU RXQ) */
    fwd.type             = DOCA_FLOW_FWD_RSS;
    fwd.rss_type         = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array = rss_queue;
    fwd.rss.nr_queues    = 1;
    fwd.rss.outer_flags  = 0;

    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, port->flow_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,    "BRIDGE_L2_PIPE");
    doca_flow_pipe_cfg_set_type(pipe_cfg,    DOCA_FLOW_PIPE_BASIC);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
    doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);  /* match={0} = wildcard */
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &port->root_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_create: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    /* Entry wildcard: cattura qualsiasi frame Ethernet */
    res = doca_flow_pipe_basic_add_entry(
        0,                /* pipe_queue */
        port->root_pipe,
        &match,           /* match={0} = qualsiasi pacchetto */
        0,                /* flags */
        NULL, NULL, NULL, /* actions, actions_mask, monitor */
        DOCA_FLOW_ENTRY_FLAGS_NO_WAIT,
        NULL,
        &port->root_entry);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_basic_add_entry: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_flow_entries_process(port->flow_port, 0, 0, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_entries_process: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    printf("[%s] DOCA Flow pronto — BASIC ROOT wildcard → RSS queue 0\n", port_label);
    return DOCA_SUCCESS;
}

/* ==========================================================================
 * CLEANUP
 * ==========================================================================
 * Teardown in ordine inverso al setup.
 * Il kernel CUDA deve essere già terminato prima di chiamare questa funzione.
 */
static void cleanup_all(struct bridge_port *ports,
                         int                 n_ports,
                         struct doca_gpu    *gdev,
                         uint64_t           *mac_table_gpu,
                         uint32_t           *gpu_exit,
                         uint64_t           *gpu_fwd,
                         uint64_t           *gpu_rx_total,
                         uint64_t           *gpu_flood,
                         uint64_t           *gpu_unicast,
                         uint64_t           *gpu_drop,
                         struct mac_flap_record *gpu_flap_ring,
                         uint32_t           *gpu_flap_head,
                         cudaStream_t        stream,
                         bool                flow_initialized)
{
    if (stream)        cudaStreamDestroy(stream);
    if (mac_table_gpu) cudaFree(mac_table_gpu);

    /* Libera variabili di sincronizzazione GPU */
    if (gpu_exit && gdev) doca_gpu_mem_free(gdev, gpu_exit);
    if (gpu_fwd  && gdev) doca_gpu_mem_free(gdev, gpu_fwd);

    /* Libera contatori diagnostici (flap detection, flood/unicast/drop) */
    if (gpu_rx_total && gdev) doca_gpu_mem_free(gdev, gpu_rx_total);
    if (gpu_flood    && gdev) doca_gpu_mem_free(gdev, gpu_flood);
    if (gpu_unicast  && gdev) doca_gpu_mem_free(gdev, gpu_unicast);
    if (gpu_drop     && gdev) doca_gpu_mem_free(gdev, gpu_drop);
    if (gpu_flap_ring && gdev) doca_gpu_mem_free(gdev, gpu_flap_ring);
    if (gpu_flap_head && gdev) doca_gpu_mem_free(gdev, gpu_flap_head);

    for (int p = 0; p < n_ports; p++) {
        if (ports[p].root_pipe) doca_flow_pipe_destroy(ports[p].root_pipe);
        if (ports[p].flow_port) doca_flow_port_stop(ports[p].flow_port);

        if (ports[p].txq_ctx) doca_ctx_stop(ports[p].txq_ctx);
        if (ports[p].txq_cpu) doca_eth_txq_destroy(ports[p].txq_cpu);

        if (ports[p].rxq_ctx)  doca_ctx_stop(ports[p].rxq_ctx);
        if (ports[p].rxq_buf && gdev)
            doca_gpu_mem_free(gdev, ports[p].rxq_buf);
        if (ports[p].rxq_cpu)  doca_eth_rxq_destroy(ports[p].rxq_cpu);
        if (ports[p].rxq_mmap) doca_mmap_destroy(ports[p].rxq_mmap);

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
        "Esempio (BF2, 2 porte):\n"
        "  sudo ip netns exec bf2 ./%s -n ad:00.0 -n ad:00.1 -g b0:00.0\n",
        prog, MAX_N_PORTS, prog);
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

    /* ── Variabili principali ─────────────────────────────────────────── */
    doca_error_t          res          = DOCA_SUCCESS;
    struct bridge_port    ports[MAX_N_PORTS];
    struct doca_dev      *all_ddevs[MAX_N_PORTS];
    struct doca_gpu      *gdev         = NULL;
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

    cudaSetDevice(cuda_id);

    /* ── 1. Apri TUTTE le NIC prima di qualsiasi altra operazione ────────
     * Necessario perché setup_port_rxq() ha bisogno dell'array completo
     * all_ddevs[] per registrare il mmap con tutte le NIC.
     */
    for (int p = 0; p < n_ports; p++) {
        res = open_nic_by_pci(nic_pci[p], &ports[p].ddev);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "Apertura NIC %s fallita\n", nic_pci[p]);
            /* Chiudi quelle già aperte */
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

    /* ── 3. Flow port per ogni porta (ID: 0, 1, ..., n_ports-1) ─────── */
    for (int p = 0; p < n_ports; p++) {
        res = flow_start_port(&ports[p], p);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 4. Crea GPU context ─────────────────────────────────────────── */
    res = doca_gpu_create(gpu_pci, &gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "doca_gpu_create (%s): %s\n", gpu_pci,
                doca_error_get_descr(res));
        goto cleanup;
    }
    printf("GPU %s (CUDA device %d) aperta\n", gpu_pci, cuda_id);

    /* ── 5. RXQ GPU per ogni porta ───────────────────────────────────────
     * Ogni RXQ viene registrata con TUTTI gli N ddev (cross-port N porte).
     */
    char port_label[32];
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_rxq(&ports[p], all_ddevs, n_ports, gdev, cuda_id, port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 6. TXQ GPU per ogni porta ──────────────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_txq(&ports[p], gdev, port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 7. DOCA Flow pipe per ogni porta ─────────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_flow(&ports[p], port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 8. Alloca MAC table (GPU memory pura: il kernel usa atomics) ─── */
    if (cudaMalloc((void **)&mac_table_gpu,
                   (size_t)MAC_TABLE_SIZE * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc MAC table fallita\n");
        goto cleanup;
    }
    cudaMemset(mac_table_gpu, 0, (size_t)MAC_TABLE_SIZE * sizeof(uint64_t));

    /* ── 9. Alloca exit_cond (GPU_CPU) ───────────────────────────────────
     * GPU_CPU: primaria in VRAM (polling veloce dal kernel), accessibile CPU.
     * Il CPU scrive cpu_exit=1 per segnalare uscita; il GPU legge gpu_exit.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit, (void **)&cpu_exit);
    if (res != DOCA_SUCCESS || !gpu_exit) {
        fprintf(stderr, "alloc exit_cond: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_exit = 0;

    /* ── 10. Alloca fwd_count (CPU_GPU) ──────────────────────────────────
     * CPU_GPU: primaria lato CPU (lettura veloce dopo sync), accessibile GPU.
     * Il GPU scrive gpu_fwd; il CPU legge cpu_fwd dopo cudaStreamSynchronize.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_fwd, (void **)&cpu_fwd);
    if (res != DOCA_SUCCESS || !gpu_fwd) {
        fprintf(stderr, "alloc fwd_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_fwd = 0;

    /* ── 10b. Alloca contatori diagnostici (CPU_GPU) ──────────────────────
     * rx_pkt_total[n_ports]: pacchetti REALMENTE ricevuti dalla RXQ, per
     * confrontarli con fwd_count — se fwd_count supera quello che è stato
     * ricevuto, è un bug SW (fantasmi); se combaciano, il traffico è reale
     * (loop di rete o flooding legittimo per FIB non convergente).
     * flood/unicast/drop: come viene classificato ogni pacchetto ricevuto.
     * flap_ring + flap_ring_head: vedi gpu_bridge.h — rilevano un MAC che
     * cambia porta, sintomo classico di un loop L2 fisico.
     */
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

    /* ── 12. Popola bridge_kernel_params ─────────────────────────────────
     *
     * rxq_mkey_cross[p][q] = mkey del buffer GPU di rxq[p], valido per NIC porta q.
     * Il kernel usa questo mkey quando scrive WQE su txq[q] che leggono da rxq[p].
     */
    kp.n_ports   = n_ports;
    kp.mac_table = mac_table_gpu;
    kp.exit_cond = gpu_exit;
    kp.fwd_count = gpu_fwd;

    kp.rx_pkt_total   = gpu_rx_total;
    kp.flood_count    = gpu_flood;
    kp.unicast_count  = gpu_unicast;
    kp.drop_count     = gpu_drop;
    kp.flap_ring      = gpu_flap_ring;
    kp.flap_ring_head = gpu_flap_head;

    for (int p = 0; p < n_ports; p++) {
        kp.rxq_gpu[p] = ports[p].rxq_gpu;
        kp.txq_gpu[p] = ports[p].txq_gpu;
        for (int q = 0; q < n_ports; q++)
            kp.rxq_mkey_cross[p][q] = ports[p].rxq_mkey_for_port[q];
    }

    /* ── 13. Lancia il kernel CUDA persistente ─────────────────────────── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\nGPU Bridge a %d porte avviato:\n", n_ports);
    for (int p = 0; p < n_ports; p++)
        printf("  porta %d: %s\n", p, nic_pci[p]);
    printf("  GPU:     %s (CUDA device %d)\n", gpu_pci, cuda_id);
    printf("  MAC table: %d slot (FNV-1a, linear probing, 8-bit port)\n",
           MAC_TABLE_SIZE);
    printf("Premi Ctrl+C per fermare.\n\n");

    res = kernel_launch_bridge(stream, &kp);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "kernel_launch_bridge: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }

    /* ── 14. Aspetta Ctrl+C, stampando statistiche live ────────────────
     * I contatori sono pubblicati dal kernel una volta per ogni giro
     * completo su tutte le porte (vedi gpu_bridge_kernel.cu), quindi qui
     * possiamo leggerli direttamente senza cudaMemcpy/sync: sono in
     * memoria CPU_GPU (primaria lato host), coerenti grazie al
     * __threadfence_system() lato GPU.
     *
     * DIAGNOSTICA: rx_tot = pacchetti REALMENTE ricevuti dalla RXQ.
     *   Se fwd > rx_tot(porta0)+rx_tot(porta1) è impossibile per costruzione
     *   (con 2 porte al più 1 egress per pacchetto): sarebbe la prova di un
     *   bug SW che "inventa" pacchetti. Se invece fwd cresce in proporzione
     *   a rx_tot, i pacchetti stanno arrivando davvero dal cavo.
     *   mac_flaps: un MAC che cambia porta ripetutamente è il sintomo
     *   classico di un loop L2 fisico nella rete (non un bug software).
     */
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

    /* ── 15. Ferma il kernel ─────────────────────────────────────────── */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit) = 1;
    cudaStreamSynchronize(stream);

    printf("\nFermato. Totale pacchetti forwardati: %lu\n", *cpu_fwd);
    printf("  Ricevuti per porta: ");
    for (int p = 0; p < n_ports; p++)
        printf("porta%d=%lu ", p, cpu_rx_total[p]);
    printf("\n  flood=%lu  unicast=%lu  drop=%lu  mac_flaps totali=%u\n",
           *cpu_flood, *cpu_unicast, *cpu_drop, *cpu_flap_head);

cleanup:
    /* gpu_exit, gpu_fwd e i contatori diagnostici vengono liberati dentro cleanup_all */
    cleanup_all(ports, n_ports, gdev, mac_table_gpu,
                gpu_exit, gpu_fwd,
                gpu_rx_total, gpu_flood, gpu_unicast, gpu_drop,
                gpu_flap_ring, gpu_flap_head,
                stream, flow_inited);
    printf("Done.\n");
    return (res == DOCA_SUCCESS) ? 0 : 1;
}
