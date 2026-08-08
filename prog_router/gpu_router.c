/*
 * gpu_router.c — setup lato CPU per il GPU L3 Router a N porte.
 *
 * Come nel bridge (prog_gpu_bridge/gpu_bridge.c), questo file si occupa
 * SOLO del "control plane": aprire i device DOCA, allocare le code RX/TX,
 * configurare DOCA Flow, caricare la tabella di routing da file e lanciare
 * il kernel CUDA persistente che fa tutto il lavoro pacchetto-per-pacchetto
 * (il "data plane", in gpu_router_kernel.cu).
 *
 * DIFFERENZE PRINCIPALI RISPETTO AL BRIDGE (prog_gpu_bridge):
 *
 *   1. Niente MAC learning / tabella FIB dinamica: la tabella di
 *      forwarding (qui chiamata FIB, coerentemente con la terminologia
 *      IP — vedi gpu_router.h) è STATICA, letta una sola volta da un file
 *      di testo (-r routes.txt) e mai più modificata mentre il kernel gira.
 *
 *   2. DOCA Flow filtra SOLO EtherType IPv4 (0x0800) all'ingresso, non
 *      tutto il traffico L2 come nel bridge: un router IP non ha nulla
 *      da fare con ARP o altri EtherType, quindi li scartiamo già in
 *      hardware (nella NIC), senza sprecare cicli GPU. Vedi setup_port_flow.
 *
 *   3. Ogni porta ha un proprio MAC address "vero" (letto dal device DOCA
 *      con doca_devinfo_get_mac_addr): un router, a differenza di un
 *      bridge trasparente, è un hop L2 a tutti gli effetti e deve
 *      presentarsi con il proprio MAC come sorgente dei pacchetti che
 *      inoltra.
 *
 * Utilizzo:
 *   sudo ip netns exec bf2 ./gpu_router \
 *       -n ad:00.0 -n ad:00.1 [-n <pci_portaN>...] -g b0:00.0 -r routes.txt
 *
 * Formato del file di rotte (vedi anche routes.example.txt):
 *   # righe che iniziano con '#' e righe vuote sono ignorate
 *   <rete>/<prefisso>  <porta_uscita>  <mac_next_hop>
 *   10.0.0.0/24         0               aa:bb:cc:dd:ee:ff
 *   0.0.0.0/0           1               11:22:33:44:55:66
 *
 * Note:
 *   - Eseguire nel netns bf2: DOCA usa libibverbs (namespace-aware)
 *   - Questo router NON implementa ARP: gli host/router adiacenti alle
 *     sue interfacce devono avere una voce ARP statica che punti al MAC
 *     reale della porta corrispondente (stampato all'avvio, vedi anche
 *     README.md).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <ctype.h>
#include <endian.h>
#include <arpa/inet.h>
#include <netinet/in.h>

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
#include <doca_flow_net.h>

#include "gpu_router.h"

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

/* Formatta un MAC a 6 byte come stringa "aa:bb:cc:dd:ee:ff". Buffer statico
 * come in gpu_bridge.c: comodo per i printf, non thread-safe (non ci serve,
 * il control plane è single-threaded). */
static const char *mac_to_str(const uint8_t mac[ETHER_ADDR_LEN])
{
    static char buf[18];
    snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return buf;
}

/* ==========================================================================
 * APERTURA DEVICE NIC
 * ==========================================================================
 * Identica al bridge: scansiona i DOCA device e apre quello con il PCI
 * address dato. doca_devinfo_is_equal_pci_addr gestisce "ad:00.0" e
 * "0000:ad:00.0" indifferentemente.
 */
static doca_error_t open_nic_by_pci(const char *pci_addr, struct doca_dev **dev, uint8_t own_mac[ETHER_ADDR_LEN])
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
            /* Il MAC address si legge dal devinfo, PRIMA di aprire il device
             * (doca_dev_as_devinfo funziona anche dopo, ma è più semplice
             * leggerlo qui mentre abbiamo ancora la lista sotto mano). */
            res = doca_devinfo_get_mac_addr(list[i], own_mac, ETHER_ADDR_LEN);
            if (res != DOCA_SUCCESS) {
                fprintf(stderr, "doca_devinfo_get_mac_addr('%s'): %s\n",
                        pci_addr, doca_error_get_descr(res));
                doca_devinfo_destroy_list(list);
                return res;
            }

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
 */
static doca_error_t flow_start_port(struct router_port *port, int port_id)
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
 * SETUP CODA DI RICEZIONE (RXQ) — identico al bridge
 * ==========================================================================
 * Il datapath GPUNetIO di ricezione non dipende da cosa il kernel CUDA
 * farà dei pacchetti: la logica di mmap cross-port (ogni buffer GPU deve
 * essere leggibile da TUTTE le NIC per il forwarding zero-copy) è la
 * stessa identica del bridge. Vedi gpu_bridge.c per i commenti estesi
 * sul "perché" di ogni chiamata; qui manteniamo lo stesso ordine di
 * operazioni, verificato funzionante su questo stesso hardware (BF2 + A30X).
 */
static doca_error_t setup_port_rxq(struct router_port *port,
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

    res = doca_eth_rxq_create(port->ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &port->rxq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_create: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_rxq_set_type(port->rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_type: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    res = doca_eth_rxq_estimate_packet_buf_size(
            DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, 0, 0,
            &cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] estimate_packet_buf_size: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }
    cyclic_buf_size = (uint32_t)ALIGN_UP(cyclic_buf_size, page_sz);

    res = doca_gpu_mem_alloc(gdev, cyclic_buf_size, page_sz, DOCA_GPU_MEM_TYPE_GPU,
                              &port->rxq_buf, NULL);
    if (res != DOCA_SUCCESS || !port->rxq_buf) {
        fprintf(stderr, "[%s] doca_gpu_mem_alloc RXQ: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    res = doca_mmap_create(&port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_create: %s\n", port_label, doca_error_get_descr(res));
        goto err_free_buf_early;
    }

    res = doca_gpu_dmabuf_fd(gdev, port->rxq_buf, cyclic_buf_size, &port->rxq_dmabuf_fd);
    if (res != DOCA_SUCCESS) {
        printf("[%s] DMABuf non disponibile, uso nvidia-peermem\n", port_label);
        res = doca_mmap_set_memrange(port->rxq_mmap, port->rxq_buf, cyclic_buf_size);
    } else {
        res = doca_mmap_set_dmabuf_memrange(port->rxq_mmap, port->rxq_dmabuf_fd,
                                             port->rxq_buf, 0, cyclic_buf_size);
    }
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] mmap set_memrange: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    res = doca_mmap_set_permissions(port->rxq_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_permissions: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* Deve precedere add_dev: per default DOCA riserva spazio per 1 solo
     * device nella struttura interna del mmap (vedi gpu_bridge.c). */
    res = doca_mmap_set_max_num_devices(port->rxq_mmap, (uint32_t)n_ports);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_max_num_devices: %s\n", port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* Registra il buffer con TUTTI gli N ddev: ogni NIC ottiene il proprio
     * mkey per fare DMA READ su questo buffer (forwarding cross-port). */
    for (int q = 0; q < n_ports; q++) {
        res = doca_mmap_add_dev(port->rxq_mmap, all_ddevs[q]);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_add_dev(porta %d): %s\n",
                    port_label, q, doca_error_get_descr(res));
            goto err_destroy_mmap;
        }
    }

    res = doca_mmap_start(port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_start: %s\n", port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    for (int q = 0; q < n_ports; q++) {
        uint32_t raw_mkey;
        res = doca_mmap_get_mkey(port->rxq_mmap, all_ddevs[q], &raw_mkey);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] doca_mmap_get_mkey(porta %d): %s\n",
                    port_label, q, doca_error_get_descr(res));
            goto err_free_buf;
        }
        /* htobe32: i WQE InfiniBand/RDMA vogliono la mkey in big-endian. */
        port->rxq_mkey_for_port[q] = htobe32(raw_mkey);
    }

    res = doca_eth_rxq_set_pkt_buf(port->rxq_cpu, port->rxq_mmap, 0, cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_pkt_buf: %s\n", port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* Pre-Hopper GPU (A30X = sm_80): abilita multicast QP workaround. */
    cudaGetDeviceProperties(&prop, cuda_id);
    if (prop.major < 9) {
        res = doca_eth_rxq_gpu_enable_mcst_qp(port->rxq_cpu);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "[%s] enable_mcst_qp: %s\n", port_label, doca_error_get_descr(res));
            goto err_free_buf;
        }
    }

    port->rxq_ctx = doca_eth_rxq_as_doca_ctx(port->rxq_cpu);
    if (!port->rxq_ctx) {
        fprintf(stderr, "[%s] doca_eth_rxq_as_doca_ctx fallito\n", port_label);
        goto err_free_buf;
    }

    res = doca_ctx_set_datapath_on_gpu(port->rxq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu RXQ: %s\n", port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    res = doca_ctx_start(port->rxq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start RXQ: %s\n", port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    res = doca_eth_rxq_get_gpu_handle(port->rxq_cpu, &port->rxq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_get_gpu_handle: %s\n", port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->rxq_ctx);
        goto err_free_buf;
    }

    printf("[%s] RXQ pronta — buf %p  size %u B  MAC %s\n",
           port_label, port->rxq_buf, cyclic_buf_size, mac_to_str(port->own_mac));
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
 * SETUP CODA DI TRASMISSIONE (TXQ) — identico al bridge
 * ==========================================================================
 */
static doca_error_t setup_port_txq(struct router_port *port,
                                    struct doca_gpu    *gdev,
                                    const char         *port_label)
{
    doca_error_t res;

    res = doca_eth_txq_create(port->ddev, MAX_SQ_DESCR_NUM, &port->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_create: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_eth_txq_set_l3_chksum_offload(port->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l3_chksum_offload: %s\n", port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_set_l4_chksum_offload(port->txq_cpu, 1);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_l4_chksum_offload: %s\n", port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_gpu_set_completion_on_gpu(port->txq_cpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] set_completion_on_gpu: %s\n", port_label, doca_error_get_descr(res));
        goto err;
    }

    port->txq_ctx = doca_eth_txq_as_doca_ctx(port->txq_cpu);
    if (!port->txq_ctx) {
        fprintf(stderr, "[%s] doca_eth_txq_as_doca_ctx fallito\n", port_label);
        goto err;
    }

    res = doca_ctx_set_datapath_on_gpu(port->txq_ctx, gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] ctx_set_datapath_on_gpu TXQ: %s\n", port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_ctx_start(port->txq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start TXQ: %s\n", port_label, doca_error_get_descr(res));
        goto err;
    }

    res = doca_eth_txq_apply_queue_id(port->txq_cpu, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] txq_apply_queue_id: %s\n", port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->txq_ctx);
        goto err;
    }

    res = doca_eth_txq_get_gpu_handle(port->txq_cpu, &port->txq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_txq_get_gpu_handle: %s\n", port_label, doca_error_get_descr(res));
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
 * SETUP DOCA FLOW — BASIC ROOT pipe, match "solo EtherType IPv4"
 * ==========================================================================
 * A differenza del bridge (match={0} = wildcard, accetta qualunque frame),
 * un router IP non ha alcun compito da svolgere su ARP, IPv6, o altri
 * EtherType: la entry della pipe qui applica un match ESPLICITO su
 * outer.eth.type == 0x0800 (IPv4). Tutto il resto (compreso ARP) finisce
 * nel "miss" della pipe (miss_fwd = DROP) e viene scartato direttamente
 * dalla NIC, senza mai raggiungere la GPU: risparmia cicli GPU E banda
 * PCIe per traffico che comunque non sapremmo instradare.
 *
 * Campo usato: struct doca_flow_match.outer.eth.type, con la costante
 * DOCA_FLOW_ETHER_TYPE_IPV4 (doca_flow_net.h) — lo stesso valore usato per
 * il match EtherType nel sample ufficiale NVIDIA "gpu_packet_processing"
 * (config_queues/flow.c). Quel sample imposta ANCHE outer.l3_type in
 * combinazione con outer.eth.type, ma lo fa su una pipe di tipo CONTROL;
 * su una pipe BASIC come questa (verificato sperimentalmente su questo
 * firmware/hardware) impostare outer.l3_type insieme a outer.eth.type fa
 * fallire doca_flow_pipe_create con DOCA_ERROR_INVALID_VALUE. Il solo
 * match su eth.type è comunque sufficiente ed equivalente ai fini pratici:
 * l'EtherType 0x0800 identifica IPv4 in modo univoco, quindi outer.l3_type
 * (che ne è ridondante) non aggiunge alcuna selettività qui.
 *
 * IMPORTANTE — implicazione pratica: questo router non risponde all'ARP.
 * Gli host/router adiacenti a ciascuna porta devono avere una voce ARP
 * statica per l'IP del router che punti al MAC reale della porta
 * (stampato all'avvio da setup_port_rxq / dal main). Vedi README.md.
 */
static doca_error_t setup_port_flow(struct router_port *port,
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

    match.outer.eth.type = htons(DOCA_FLOW_ETHER_TYPE_IPV4);

    res = doca_eth_rxq_apply_queue_id(port->rxq_cpu, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_apply_queue_id: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    fwd.type             = DOCA_FLOW_FWD_RSS;
    fwd.rss_type         = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array = rss_queue;
    fwd.rss.nr_queues    = 1;
    fwd.rss.outer_flags  = 0;

    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, port->flow_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,    "ROUTER_L3_PIPE");
    doca_flow_pipe_cfg_set_type(pipe_cfg,    DOCA_FLOW_PIPE_BASIC);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
    doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &port->root_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_create: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_flow_pipe_basic_add_entry(
        0, port->root_pipe, &match, 0,
        NULL, NULL, NULL,
        DOCA_FLOW_ENTRY_FLAGS_NO_WAIT,
        NULL, &port->root_entry);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_basic_add_entry: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    res = doca_flow_entries_process(port->flow_port, 0, 0, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_entries_process: %s\n", port_label, doca_error_get_descr(res));
        return res;
    }

    printf("[%s] DOCA Flow pronto — BASIC ROOT match EtherType IPv4 → RSS queue 0 (resto: DROP)\n", port_label);
    return DOCA_SUCCESS;
}

/* ==========================================================================
 * TABELLA DI ROUTING — parsing del file e costruzione della FIB
 * ==========================================================================
 */

/* build_mask_be: costruisce la maschera di rete per un prefisso CIDR
 * (0..32), già in NETWORK BYTE ORDER (vedi commento su route_entry in
 * gpu_router.h: la teniamo così per evitare byte-swap nel kernel CUDA).
 * Il caso prefix_len==0 è gestito a parte perché "1u << 32" è undefined
 * behaviour in C (shift pari alla larghezza del tipo). */
static uint32_t build_mask_be(int prefix_len)
{
    if (prefix_len <= 0)
        return 0;
    if (prefix_len >= 32)
        return htonl(0xFFFFFFFFu);
    return htonl(0xFFFFFFFFu << (32 - prefix_len));
}

/* parse_mac_str: converte "aa:bb:cc:dd:ee:ff" in 6 byte. Ritorna true se
 * il parsing ha prodotto esattamente 6 valori esadecimali validi. */
static bool parse_mac_str(const char *str, uint8_t mac[ETHER_ADDR_LEN])
{
    unsigned int b[ETHER_ADDR_LEN];
    int n = sscanf(str, "%2x:%2x:%2x:%2x:%2x:%2x",
                   &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]);
    if (n != ETHER_ADDR_LEN)
        return false;
    for (int i = 0; i < ETHER_ADDR_LEN; i++)
        mac[i] = (uint8_t)b[i];
    return true;
}

/* route_cmp_by_prefixlen_desc: comparatore per qsort. Ordina le rotte dal
 * prefisso PIÙ LUNGO al più corto: questo è ciò che rende corretta la
 * scansione lineare fib_lookup() nel kernel CUDA (il primo match trovato,
 * scandendo in quest'ordine, è per costruzione il Longest Prefix Match —
 * vedi la spiegazione estesa in gpu_router_kernel.cu). */
static int route_cmp_by_prefixlen_desc(const void *a, const void *b)
{
    const struct route_entry *ra = (const struct route_entry *)a;
    const struct route_entry *rb = (const struct route_entry *)b;
    return (int)rb->prefix_len - (int)ra->prefix_len;
}

/* load_routes_file: legge il file di rotte riga per riga.
 *
 * Formato di ogni riga non vuota/non commento:
 *   <rete>/<prefisso>  <porta_uscita>  <mac_next_hop>
 * Esempio:
 *   10.0.0.0/24  0  aa:bb:cc:dd:ee:ff
 *
 * Ritorna DOCA_SUCCESS con *n_routes valorizzato, oppure un errore con un
 * messaggio che indica il numero di riga (fondamentale per un file di
 * configurazione: senza questo, un typo diventa un mistero da debuggare
 * a tentoni).
 */
static doca_error_t load_routes_file(const char *path,
                                      struct route_entry *rib,
                                      int *n_routes,
                                      int n_ports)
{
    FILE *f;
    char line[256];
    int lineno = 0;
    int count = 0;

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Impossibile aprire il file di rotte '%s'\n", path);
        return DOCA_ERROR_NOT_FOUND;
    }

    while (fgets(line, sizeof(line), f)) {
        lineno++;

        /* Salta spazi iniziali, poi righe vuote o commenti ('#'). */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '#')
            continue;

        if (count >= MAX_ROUTES) {
            fprintf(stderr, "routes.txt:%d: troppe rotte (max %d), riga ignorata\n",
                    lineno, MAX_ROUTES);
            continue;
        }

        char cidr[64], macstr[64];
        int egress_port;
        int n = sscanf(p, "%63s %d %63s", cidr, &egress_port, macstr);
        if (n != 3) {
            fprintf(stderr, "routes.txt:%d: formato non valido (attesi 3 campi: rete/prefisso porta mac)\n",
                    lineno);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }

        char *slash = strchr(cidr, '/');
        if (!slash) {
            fprintf(stderr, "routes.txt:%d: manca '/<prefisso>' in '%s'\n", lineno, cidr);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }
        *slash = '\0';
        int prefix_len = atoi(slash + 1);
        if (prefix_len < 0 || prefix_len > 32) {
            fprintf(stderr, "routes.txt:%d: prefisso /%d fuori range [0,32]\n", lineno, prefix_len);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }

        struct in_addr net_addr;
        if (inet_pton(AF_INET, cidr, &net_addr) != 1) {
            fprintf(stderr, "routes.txt:%d: indirizzo IP non valido '%s'\n", lineno, cidr);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }

        if (egress_port < 0 || egress_port >= n_ports) {
            fprintf(stderr, "routes.txt:%d: porta di uscita %d fuori range [0,%d]\n",
                    lineno, egress_port, n_ports - 1);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }

        uint8_t next_hop_mac[ETHER_ADDR_LEN];
        if (!parse_mac_str(macstr, next_hop_mac)) {
            fprintf(stderr, "routes.txt:%d: MAC non valido '%s'\n", lineno, macstr);
            fclose(f);
            return DOCA_ERROR_INVALID_VALUE;
        }

        uint32_t mask_be = build_mask_be(prefix_len);
        uint32_t network_be = net_addr.s_addr & mask_be;
        if (network_be != net_addr.s_addr) {
            char norm[INET_ADDRSTRLEN];
            struct in_addr tmp = { .s_addr = network_be };
            inet_ntop(AF_INET, &tmp, norm, sizeof(norm));
            printf("routes.txt:%d: attenzione — %s/%d ha bit host non a zero, normalizzato a %s/%d\n",
                   lineno, cidr, prefix_len, norm, prefix_len);
        }

        rib[count].network     = network_be;
        rib[count].mask        = mask_be;
        rib[count].prefix_len  = (uint8_t)prefix_len;
        rib[count].egress_port = (uint8_t)egress_port;
        memcpy(rib[count].next_hop_mac, next_hop_mac, ETHER_ADDR_LEN);
        count++;
    }

    fclose(f);

    if (count == 0) {
        fprintf(stderr, "routes.txt: nessuna rotta valida trovata in '%s'\n", path);
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Compila la RIB in FIB: ordina per prefisso decrescente, così la
     * scansione lineare nel kernel trova sempre il Longest Prefix Match
     * per prima. Vedi route_cmp_by_prefixlen_desc e gpu_router_kernel.cu. */
    qsort(rib, (size_t)count, sizeof(struct route_entry), route_cmp_by_prefixlen_desc);

    *n_routes = count;
    return DOCA_SUCCESS;
}

/* ==========================================================================
 * CLEANUP
 * ==========================================================================
 * Teardown in ordine inverso al setup. Il kernel CUDA deve essere già
 * terminato prima di chiamare questa funzione.
 */
static void cleanup_all(struct router_port *ports,
                         int                 n_ports,
                         struct doca_gpu    *gdev,
                         struct route_entry *fib_gpu,
                         uint32_t           *gpu_exit,
                         uint64_t           *gpu_fwd,
                         uint64_t           *gpu_rx_total,
                         uint64_t           *gpu_drop_no_route,
                         uint64_t           *gpu_drop_ttl_expired,
                         uint64_t           *gpu_drop_malformed,
                         cudaStream_t        stream,
                         bool                flow_initialized)
{
    if (stream)  cudaStreamDestroy(stream);
    if (fib_gpu) cudaFree(fib_gpu);

    if (gpu_exit && gdev) doca_gpu_mem_free(gdev, gpu_exit);
    if (gpu_fwd  && gdev) doca_gpu_mem_free(gdev, gpu_fwd);

    if (gpu_rx_total       && gdev) doca_gpu_mem_free(gdev, gpu_rx_total);
    if (gpu_drop_no_route  && gdev) doca_gpu_mem_free(gdev, gpu_drop_no_route);
    if (gpu_drop_ttl_expired && gdev) doca_gpu_mem_free(gdev, gpu_drop_ttl_expired);
    if (gpu_drop_malformed && gdev) doca_gpu_mem_free(gdev, gpu_drop_malformed);

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
        "-g <pci_gpu> -r <file_rotte> [-i <cuda_id>]\n"
        "\n"
        "  -n  PCI address di una porta NIC (ripetere per ogni porta, min 2, max %d)\n"
        "  -g  PCI address della GPU\n"
        "  -r  file di testo con le rotte statiche (vedi routes.example.txt)\n"
        "  -i  CUDA device index (default: 0)\n"
        "\n"
        "Esempio (BF2, 2 porte):\n"
        "  sudo ip netns exec bf2 ./%s -n ad:00.0 -n ad:00.1 -g b0:00.0 -r routes.txt\n",
        prog, MAX_N_PORTS, prog);
}

/* ==========================================================================
 * MAIN
 * ==========================================================================
 */
int main(int argc, char *argv[])
{
    char nic_pci[MAX_N_PORTS][DOCA_DEVINFO_PCI_ADDR_SIZE];
    char gpu_pci[64]   = {0};
    char routes_path[256] = {0};
    int  cuda_id    = 0;
    int  n_ports    = 0;
    int  opt;

    memset(nic_pci, 0, sizeof(nic_pci));

    while ((opt = getopt(argc, argv, "n:g:i:r:h")) != -1) {
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
        case 'r':
            strncpy(routes_path, optarg, sizeof(routes_path) - 1);
            break;
        case 'i':
            cuda_id = atoi(optarg);
            break;
        case 'h': default:
            usage(argv[0]); return 1;
        }
    }

    if (n_ports < 2 || !gpu_pci[0] || !routes_path[0]) {
        fprintf(stderr, "Errore: servono almeno 2 flag -n, 1 flag -g e 1 flag -r.\n");
        usage(argv[0]);
        return 1;
    }

    /* ── Variabili principali ─────────────────────────────────────────── */
    doca_error_t          res          = DOCA_SUCCESS;
    struct router_port    ports[MAX_N_PORTS];
    struct doca_dev      *all_ddevs[MAX_N_PORTS];
    struct doca_gpu      *gdev         = NULL;
    struct route_entry    rib[MAX_ROUTES];
    int                   n_routes     = 0;
    struct route_entry   *fib_gpu      = NULL;
    uint32_t             *gpu_exit     = NULL;
    uint32_t             *cpu_exit     = NULL;
    uint64_t             *gpu_fwd      = NULL;
    uint64_t             *cpu_fwd      = NULL;
    uint64_t             *gpu_rx_total = NULL;
    uint64_t             *cpu_rx_total = NULL;
    uint64_t             *gpu_drop_no_route    = NULL;
    uint64_t             *cpu_drop_no_route    = NULL;
    uint64_t             *gpu_drop_ttl_expired = NULL;
    uint64_t             *cpu_drop_ttl_expired = NULL;
    uint64_t             *gpu_drop_malformed   = NULL;
    uint64_t             *cpu_drop_malformed   = NULL;
    cudaStream_t          stream       = NULL;
    bool                  flow_inited  = false;
    struct router_kernel_params kp     = {0};

    memset(ports,     0, sizeof(ports));
    memset(all_ddevs, 0, sizeof(all_ddevs));
    memset(rib, 0, sizeof(rib));

    cudaSetDevice(cuda_id);

    /* ── 1. Apri TUTTE le NIC prima di qualsiasi altra operazione ────────
     * (leggendo anche il MAC di ciascuna, servirà a riscrivere il MAC
     * sorgente dei pacchetti instradati — vedi open_nic_by_pci). */
    for (int p = 0; p < n_ports; p++) {
        res = open_nic_by_pci(nic_pci[p], &ports[p].ddev, ports[p].own_mac);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "Apertura NIC %s fallita\n", nic_pci[p]);
            for (int j = 0; j < p; j++) doca_dev_close(ports[j].ddev);
            return 1;
        }
        all_ddevs[p] = ports[p].ddev;
        printf("NIC porta %d aperta: %s  MAC %s\n", p, nic_pci[p], mac_to_str(ports[p].own_mac));
    }

    /* ── 2. Carica la tabella di routing (RIB -> FIB, ordinata) ─────────
     * Fatto presto, PRIMA di allocare risorse GPU costose: se il file di
     * rotte ha un errore, vogliamo uscire subito senza aver aperto nulla
     * di pesante (fail fast). */
    res = load_routes_file(routes_path, rib, &n_routes, n_ports);
    if (res != DOCA_SUCCESS) {
        for (int p = 0; p < n_ports; p++) doca_dev_close(ports[p].ddev);
        return 1;
    }
    printf("Caricate %d rotte da '%s' (ordinate per prefisso decrescente):\n", n_routes, routes_path);
    for (int i = 0; i < n_routes; i++) {
        char netstr[INET_ADDRSTRLEN];
        struct in_addr tmp = { .s_addr = rib[i].network };
        inet_ntop(AF_INET, &tmp, netstr, sizeof(netstr));
        printf("  %-18s /%2u -> porta %u  next-hop %s\n",
               netstr, rib[i].prefix_len, rib[i].egress_port, mac_to_str(rib[i].next_hop_mac));
    }

    /* ── 3. Init globale DOCA Flow ────────────────────────────────────── */
    res = flow_global_init();
    if (res != DOCA_SUCCESS) goto cleanup;
    flow_inited = true;

    /* ── 4. Flow port per ogni porta (ID: 0, 1, ..., n_ports-1) ─────── */
    for (int p = 0; p < n_ports; p++) {
        res = flow_start_port(&ports[p], p);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 5. Crea GPU context ─────────────────────────────────────────── */
    res = doca_gpu_create(gpu_pci, &gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "doca_gpu_create (%s): %s\n", gpu_pci, doca_error_get_descr(res));
        goto cleanup;
    }
    printf("GPU %s (CUDA device %d) aperta\n", gpu_pci, cuda_id);

    /* ── 6. RXQ GPU per ogni porta (cross-port, come nel bridge) ────────── */
    char port_label[32];
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_rxq(&ports[p], all_ddevs, n_ports, gdev, cuda_id, port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 7. TXQ GPU per ogni porta ──────────────────────────────────── */
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_txq(&ports[p], gdev, port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 8. DOCA Flow pipe per ogni porta (match solo IPv4) ───────────── */
    for (int p = 0; p < n_ports; p++) {
        snprintf(port_label, sizeof(port_label), "porta %d", p);
        res = setup_port_flow(&ports[p], port_label);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 9. Carica la FIB in memoria GPU ──────────────────────────────
     * A differenza della mac_table del bridge (scritta continuamente dal
     * kernel con atomicCAS/atomicExch), la FIB è statica: un singolo
     * cudaMemcpy basta, nessun bisogno di allocazione "GPU_CPU" o di
     * atomici nel kernel (vedi commenti su router_kernel_params in
     * gpu_router.h). */
    if (cudaMalloc((void **)&fib_gpu, (size_t)n_routes * sizeof(struct route_entry)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc FIB fallita\n");
        goto cleanup;
    }
    if (cudaMemcpy(fib_gpu, rib, (size_t)n_routes * sizeof(struct route_entry),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy FIB fallita\n");
        goto cleanup;
    }

    /* ── 10. Alloca exit_cond (GPU_CPU): il CPU segnala l'uscita al kernel ── */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit, (void **)&cpu_exit);
    if (res != DOCA_SUCCESS || !gpu_exit) {
        fprintf(stderr, "alloc exit_cond: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_exit = 0;

    /* ── 11. Contatori diagnostici (CPU_GPU: scritti dalla GPU, letti dalla
     * CPU senza sync esplicito grazie a __threadfence_system nel kernel) ── */
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
                              (void **)&gpu_drop_no_route, (void **)&cpu_drop_no_route);
    if (res != DOCA_SUCCESS || !gpu_drop_no_route) {
        fprintf(stderr, "alloc drop_no_route: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_drop_no_route = 0;

    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_drop_ttl_expired, (void **)&cpu_drop_ttl_expired);
    if (res != DOCA_SUCCESS || !gpu_drop_ttl_expired) {
        fprintf(stderr, "alloc drop_ttl_expired: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_drop_ttl_expired = 0;

    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_drop_malformed, (void **)&cpu_drop_malformed);
    if (res != DOCA_SUCCESS || !gpu_drop_malformed) {
        fprintf(stderr, "alloc drop_malformed: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_drop_malformed = 0;

    /* ── 12. CUDA stream ─────────────────────────────────────────────── */
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags fallita\n");
        goto cleanup;
    }

    /* ── 13. Popola router_kernel_params ──────────────────────────────── */
    kp.n_ports  = n_ports;
    kp.fib      = fib_gpu;
    kp.fib_size = (uint32_t)n_routes;
    kp.exit_cond = gpu_exit;
    kp.fwd_count = gpu_fwd;

    kp.rx_pkt_total      = gpu_rx_total;
    kp.drop_no_route     = gpu_drop_no_route;
    kp.drop_ttl_expired  = gpu_drop_ttl_expired;
    kp.drop_malformed    = gpu_drop_malformed;

    for (int p = 0; p < n_ports; p++) {
        kp.rxq_gpu[p] = ports[p].rxq_gpu;
        kp.txq_gpu[p] = ports[p].txq_gpu;
        memcpy(kp.port_mac[p], ports[p].own_mac, ETHER_ADDR_LEN);
        for (int q = 0; q < n_ports; q++)
            kp.rxq_mkey_cross[p][q] = ports[p].rxq_mkey_for_port[q];
    }

    /* ── 14. Lancia il kernel CUDA persistente ─────────────────────────── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\nGPU Router a %d porte avviato:\n", n_ports);
    for (int p = 0; p < n_ports; p++)
        printf("  porta %d: %s  MAC %s\n", p, nic_pci[p], mac_to_str(ports[p].own_mac));
    printf("  GPU:     %s (CUDA device %d)\n", gpu_pci, cuda_id);
    printf("  FIB:     %d rotte (LPM per scansione lineare ordinata)\n", n_routes);
    printf("Premi Ctrl+C per fermare.\n\n");

    res = kernel_launch_router(stream, &kp);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "kernel_launch_router: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }

    /* ── 15. Aspetta Ctrl+C, stampando statistiche live ────────────────
     * Stesso schema del bridge: i contatori sono pubblicati dal kernel una
     * volta per ogni giro completo su tutte le porte, quindi si leggono
     * direttamente da memoria CPU_GPU senza cudaMemcpy/sync espliciti. */
    while (!DOCA_GPUNETIO_VOLATILE(g_force_quit)) {
        usleep(500000);

        uint64_t rx_sum = 0;
        for (int p = 0; p < n_ports; p++)
            rx_sum += cpu_rx_total[p];

        printf("[live] rx_tot=%lu (", rx_sum);
        for (int p = 0; p < n_ports; p++)
            printf("porta%d=%lu%s", p, cpu_rx_total[p], p + 1 < n_ports ? " " : "");
        printf(")  fwd=%lu  drop_no_route=%lu  drop_ttl_expired=%lu  drop_malformed=%lu\n",
               *cpu_fwd, *cpu_drop_no_route, *cpu_drop_ttl_expired, *cpu_drop_malformed);

        fflush(stdout);
    }

    /* ── 16. Ferma il kernel ─────────────────────────────────────────── */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit) = 1;
    cudaStreamSynchronize(stream);

    printf("\nFermato. Totale pacchetti instradati: %lu\n", *cpu_fwd);
    printf("  Ricevuti per porta: ");
    for (int p = 0; p < n_ports; p++)
        printf("porta%d=%lu ", p, cpu_rx_total[p]);
    printf("\n  drop_no_route=%lu  drop_ttl_expired=%lu  drop_malformed=%lu\n",
           *cpu_drop_no_route, *cpu_drop_ttl_expired, *cpu_drop_malformed);

cleanup:
    cleanup_all(ports, n_ports, gdev, fib_gpu,
                gpu_exit, gpu_fwd,
                gpu_rx_total, gpu_drop_no_route, gpu_drop_ttl_expired, gpu_drop_malformed,
                stream, flow_inited);
    printf("Done.\n");
    return (res == DOCA_SUCCESS) ? 0 : 1;
}
