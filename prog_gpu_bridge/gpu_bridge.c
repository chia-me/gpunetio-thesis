/*
 * gpu_bridge.c — setup lato CPU per il GPU L2 Bridge.
 *
 * Questo file gestisce tutto ciò che riguarda il CPU:
 *   - apertura dei device DOCA (NIC e GPU)
 *   - allocazione delle code di ricezione (RXQ) e trasmissione (TXQ)
 *   - configurazione di DOCA Flow per lo steering dei pacchetti
 *   - allocazione della MAC table e delle variabili di sincronizzazione
 *   - lancio del kernel CUDA
 *   - attesa di Ctrl+C e cleanup
 *
 * Il kernel CUDA (gpu_bridge_kernel.cu) si occupa invece del datapath
 * real-time: ricezione pacchetti, MAC learning, FIB lookup, forward/drop.
 *
 * Utilizzo:
 *   sudo ip netns exec bf2 ./gpu_bridge \
 *       -n ad:00.0 -n ad:00.1 -g b0:00.0
 *
 * Note sull'ambiente:
 *   - La BF2 deve essere nel network namespace "bf2" (ip netns exec bf2)
 *     perché DOCA usa libibverbs per scoprire i device, e libibverbs è
 *     namespace-aware: i device RDMA della BF2 sono visibili solo nel
 *     namespace dove vivono le interfacce di rete corrispondenti.
 *   - Il traffico test arriva sempre dall'Intel 810 verso la BF2
 *     (l'Intel 810 non è gestita da DOCA, è nel namespace di default).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <endian.h>   /* htobe32 */

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

/* ── Variabile globale per segnale Ctrl+C ─────────────────────────────── */
static volatile bool g_force_quit = false;

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
        DOCA_GPUNETIO_VOLATILE(g_force_quit) = true;
}

/* ── Utility: page size del sistema ──────────────────────────────────── */
static size_t system_page_size(void)
{
    long ret = sysconf(_SC_PAGESIZE);
    return (ret <= 0) ? 4096UL : (size_t)ret;
}

/* ==========================================================================
 * APERTURA DEVICE NIC
 * ==========================================================================
 *
 * Scansiona tutti i DOCA device disponibili e apre quello con l'indirizzo
 * PCI specificato.
 *
 * Perché doca_devinfo_is_equal_pci_addr invece di strcmp?
 * DOCA internamente può rappresentare l'indirizzo PCI con o senza il dominio
 * (es. "0000:ad:00.0" vs "ad:00.0"). Questa funzione confronta correttamente
 * entrambe le forme, mentre strcmp fallirebbe con il dominio.
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
 *
 * DOCA Flow è il sistema di hardware packet steering della BF2.
 * Permette di configurare regole nella NIC per instradare i pacchetti:
 * - verso code specifiche (RSS → GPU RXQ)
 * - verso altre porte
 * - verso il drop
 *
 * doca_flow_init viene chiamato UNA SOLA VOLTA per tutta l'applicazione.
 * Ogni porta fisica viene poi avviata con doca_flow_port_start.
 *
 * "vnf": modalità Virtual Network Function — adatta per bridge/router.
 * pipe_queues=1: una sola pipe queue (per semplicità; aumentabile per scaling).
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
 *
 * Ogni porta fisica della BF2 ha un port_id univoco in DOCA Flow (0 o 1).
 * Dopo questa chiamata, DOCA Flow è pronto a ricevere configurazioni
 * (pipe, entry) per questa porta.
 *
 * port_id: deve essere univoco per porta. Usiamo 0 per ad:00.0, 1 per ad:00.1.
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
 *
 * Crea e avvia una RXQ GPU per la porta specificata.
 * Questo è il passo più complesso: alloca memoria GPU, la registra con
 * ENTRAMBE le NIC (per il forwarding cross-port zero-copy) e avvia il contesto.
 *
 * Parametri:
 *   port:       la struttura della porta per cui creare la RXQ
 *   other_ddev: il device DOCA dell'ALTRA porta (necessario per cross-port)
 *   gdev:       il GPU context DOCA (la A30X)
 *   cuda_id:    indice CUDA del device GPU
 *   port_label: stringa per i messaggi di log ("porta 0" / "porta 1")
 *
 * REGISTRAZIONE CROSS-PORT:
 *   Il buffer GPU di questa porta viene registrato in due PD RDMA:
 *     1. doca_mmap_add_dev(mmap, port->ddev) → la NIC di questa porta
 *        può fare DMA WRITE nel buffer (per ricevere pacchetti)
 *     2. doca_mmap_add_dev(mmap, other_ddev) → la NIC dell'altra porta
 *        può fare DMA READ dal buffer (per trasmettere pacchetti zero-copy)
 *   Poi doca_mmap_get_mkey ci dà la chiave hardware per ciascun PD.
 */
static doca_error_t setup_port_rxq(struct bridge_port *port,
                                    struct doca_dev   *other_ddev,
                                    struct doca_gpu   *gdev,
                                    int                cuda_id,
                                    const char        *port_label)
{
    doca_error_t res;
    uint32_t cyclic_buf_size = 0;
    size_t page_sz = system_page_size();
    struct cudaDeviceProp prop;
    uint32_t mkey_self, mkey_other;

    /* ── Crea la RXQ con tipo CYCLIC ─────────────────────────────────────
     * CYCLIC = ring buffer ciclico. La NIC scrive nel prossimo slot
     * disponibile e wrappa intorno quando arriva alla fine.
     * MAX_PKT_NUM = 16384 slot, MAX_PKT_SIZE = 2048 byte per slot.
     */
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

    /* ── Calcola dimensione buffer necessaria ────────────────────────────
     * DOCA calcola la dimensione esatta del buffer ciclico tenendo conto
     * di header interni, padding e allineamenti della NIC.
     */
    res = doca_eth_rxq_estimate_packet_buf_size(
            DOCA_ETH_RXQ_TYPE_CYCLIC,
            0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, 0, 0,
            &cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] estimate_packet_buf_size: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    /* Allinea alla page size (richiesto da doca_gpu_mem_alloc) */
    cyclic_buf_size = (uint32_t)ALIGN_UP(cyclic_buf_size, page_sz);
    port->rxq_buf_size = cyclic_buf_size;

    /* ── Crea il mmap che descrive il buffer GPU alla NIC ────────────────
     * Il mmap è l'oggetto DOCA che registra una regione di memoria GPU
     * con una o più NIC tramite RDMA (simile a ibv_reg_mr in libibverbs).
     */
    res = doca_mmap_create(&port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_create: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_rxq;
    }

    /* ── Registra il mmap con ENTRAMBE le NIC (cross-port) ───────────────
     *
     * Questo è il punto chiave per il forwarding zero-copy cross-porta:
     *
     * 1) port->ddev (questa porta): la NIC può fare DMA WRITE nel buffer.
     *    Quando un pacchetto arriva su questa porta, la NIC lo scrive
     *    direttamente in rxq_buf senza passare per la CPU.
     *
     * 2) other_ddev (l'altra porta): la NIC può fare DMA READ dal buffer.
     *    Quando il kernel CUDA decide di forwardare un pacchetto, scrive
     *    un WQE sulla TXQ dell'altra porta che punta a rxq_buf di questa.
     *    La NIC dell'altra porta legge i dati direttamente dalla GPU.
     *
     * NOTA: doca_mmap_add_dev deve essere chiamato PRIMA di doca_mmap_start.
     * Dopo start, la lista dei device è fissa.
     */
    res = doca_mmap_add_dev(port->rxq_mmap, port->ddev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_add_dev (self): %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    res = doca_mmap_add_dev(port->rxq_mmap, other_ddev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_add_dev (other): %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* ── Alloca buffer GPU (memoria A30X) ────────────────────────────────
     * DOCA_GPU_MEM_TYPE_GPU: memoria esclusivamente GPU.
     * La NIC accede tramite DMA (peer-to-peer via PCIe), non tramite CPU.
     * page_sz come alignment: richiesto da DOCA GPUNetIO.
     */
    res = doca_gpu_mem_alloc(gdev, cyclic_buf_size, page_sz,
                              DOCA_GPU_MEM_TYPE_GPU,
                              &port->rxq_buf, NULL);
    if (res != DOCA_SUCCESS || !port->rxq_buf) {
        fprintf(stderr, "[%s] doca_gpu_mem_alloc RXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_destroy_mmap;
    }

    /* ── Configura il mmap: prova DMABuf, fallback su nvidia-peermem ─────
     *
     * DMABuf: metodo moderno (kernel >= 5.12) per peer-to-peer GPU↔NIC.
     *   Espone la memoria GPU come file descriptor, usato da libibverbs
     *   per registrarla nel PD della NIC senza nvidia-peermem.
     *
     * nvidia-peermem: metodo tradizionale, un kernel module che permette
     *   alla NIC di accedere alla memoria GPU tramite RDMA peer-to-peer.
     *   Richiede che il modulo gdrdrv sia caricato (gdrcopy).
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
        goto err_free_buf;
    }

    res = doca_mmap_set_permissions(port->rxq_mmap,
                                     DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_set_permissions: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Avvia il mmap: registra il buffer in entrambi i PD RDMA ─────────
     * Questo è il momento in cui DOCA effettivamente chiama ibv_reg_mr
     * per ogni device aggiunto con doca_mmap_add_dev.
     * Dopo questo punto, entrambe le NIC hanno una chiave hardware
     * per accedere al buffer GPU.
     */
    res = doca_mmap_start(port->rxq_mmap);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_start: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Ottieni i mkey per ciascun device ───────────────────────────────
     *
     * doca_mmap_get_mkey: restituisce la chiave hardware (LKEY in InfiniBand)
     * per il device specificato. Questa chiave è inclusa nei WQE della TXQ
     * per autorizzare la NIC a fare DMA dal buffer.
     *
     * rxq_mkey_for_own_txq: chiave per la NIC di questa porta.
     *   Usata se si volesse fare loopback (TX sulla stessa porta RX).
     *   Nel bridge normale non è usata, ma la memorizziamo per completezza.
     *
     * rxq_mkey_for_other_txq: chiave per la NIC dell'ALTRA porta.
     *   QUESTO è il mkey che il kernel CUDA usa nei WQE di txq dell'altra porta
     *   per il forwarding zero-copy cross-port.
     *
     * htobe32: la NIC si aspetta il mkey in network byte order (big-endian).
     */
    res = doca_mmap_get_mkey(port->rxq_mmap, port->ddev, &mkey_self);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_get_mkey (self): %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }
    port->rxq_mkey_for_own_txq = htobe32(mkey_self);

    res = doca_mmap_get_mkey(port->rxq_mmap, other_ddev, &mkey_other);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_mmap_get_mkey (other): %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }
    port->rxq_mkey_for_other_txq = htobe32(mkey_other);

    /* ── Collega buffer alla RXQ ─────────────────────────────────────── */
    res = doca_eth_rxq_set_pkt_buf(port->rxq_cpu, port->rxq_mmap, 0, cyclic_buf_size);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_set_pkt_buf: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Pre-Hopper GPU (A30X = sm_80): abilita multicast QP ────────────
     * Le GPU pre-Hopper (compute capability < 9.0) non supportano
     * direttamente il QP (Queue Pair) di ricezione normale.
     * Il "multicast QP" è un workaround che usa un QP multicast per
     * consegnare i pacchetti alla GPU. Richiesto per A30X (sm_80).
     * Le GPU Hopper+ (sm_90, es. H100) non ne hanno bisogno.
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

    /* ── Imposta la GPU come destinazione del datapath ───────────────────
     * Questo dice a DOCA: "il processing di questa RXQ avviene sulla GPU".
     * Abilitato questo, la NIC DMA direttamente in GPU memory.
     */
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

    /* ── Avvia il contesto: la RXQ è ora operativa ─────────────────────
     * Dopo questa chiamata, la NIC può iniziare a DMA pacchetti nel buffer.
     * Da questo momento la RXQ è hardware-active.
     */
    res = doca_ctx_start(port->rxq_ctx);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_ctx_start RXQ: %s\n",
                port_label, doca_error_get_descr(res));
        goto err_free_buf;
    }

    /* ── Ottieni l'handle GPU da passare al kernel CUDA ─────────────────
     * rxq_gpu è un puntatore a struttura in GPU memory.
     * Il kernel CUDA lo usa nelle chiamate doca_gpu_dev_eth_rxq_recv.
     */
    res = doca_eth_rxq_get_gpu_handle(port->rxq_cpu, &port->rxq_gpu);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_get_gpu_handle: %s\n",
                port_label, doca_error_get_descr(res));
        doca_ctx_stop(port->rxq_ctx);
        goto err_free_buf;
    }

    printf("[%s] RXQ pronta — buf %p  size %u  mkey_self 0x%08x  mkey_other 0x%08x\n",
           port_label, port->rxq_buf, cyclic_buf_size,
           port->rxq_mkey_for_own_txq, port->rxq_mkey_for_other_txq);
    return DOCA_SUCCESS;

err_free_buf:
    doca_gpu_mem_free(gdev, port->rxq_buf);
    port->rxq_buf = NULL;
err_destroy_mmap:
    doca_mmap_destroy(port->rxq_mmap);
    port->rxq_mmap = NULL;
err_destroy_rxq:
    doca_eth_rxq_destroy(port->rxq_cpu);
    port->rxq_cpu = NULL;
    return DOCA_ERROR_BAD_STATE;
}

/* ==========================================================================
 * SETUP CODA DI TRASMISSIONE (TXQ)
 * ==========================================================================
 *
 * Crea e avvia una TXQ GPU per la porta specificata.
 * La TXQ NON ha un proprio buffer dati: il kernel CUDA scrive WQE che
 * puntano al buffer rxq dell'altra porta (zero-copy cross-port).
 *
 * Opzioni abilitate:
 *   L3 checksum offload: la NIC ricalcola il checksum IP automaticamente.
 *   L4 checksum offload: la NIC ricalcola il checksum TCP/UDP automaticamente.
 *   GPU completion: le CQE vengono scritte in GPU memory (non CPU), così
 *     il kernel CUDA può fare polling direttamente senza coinvolgere la CPU.
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

    /* La NIC ricalcola i checksum L3 (IP) e L4 (TCP/UDP) in hardware.
     * Il kernel CUDA non deve calcolare nulla: il bridge può forwardare
     * anche senza conoscere il protocollo L4. */
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

    /* Le CQE vengono scritte in GPU memory: il kernel CUDA fa polling
     * direttamente senza passare dalla CPU (latenza inferiore). */
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

    /* Collega la TXQ alla queue 0 di DOCA Flow per questa porta.
     * Necessario per la corretta gestione delle risorse hardware. */
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
 * SETUP DOCA FLOW — BASIC ROOT PIPE PER IL BRIDGE
 * ==========================================================================
 *
 * Crea una singola BASIC ROOT pipe per questa porta che cattura TUTTO
 * il traffico L2 (qualsiasi ethertype) e lo invia alla GPU RXQ (queue 0).
 *
 * PERCHÉ BASIC ROOT E NON CONTROL ROOT?
 *   Il forwarder UDP usa un CONTROL pipe root perché deve distinguere
 *   il traffico (solo UDP su certa porta). Il bridge NON distingue:
 *   vuole TUTTO il traffico L2 (IPv4, ARP, IPv6, ecc.).
 *
 *   Con una BASIC ROOT pipe e match template = {0} (tutti i campi a zero),
 *   DOCA Flow interpreta ogni campo come "wildcard" = non filtrare su questo.
 *   La singola entry con match={0} cattura qualsiasi pacchetto.
 *
 * CHIAMATA ORDINE:
 *   Deve essere chiamata DOPO setup_port_rxq (perché usa port->rxq_cpu)
 *   e DOPO flow_start_port (perché usa port->flow_port).
 */
static doca_error_t setup_port_flow(struct bridge_port *port,
                                     const char         *port_label)
{
    struct doca_flow_match    match    = {0};  /* wildcard su tutti i campi */
    struct doca_flow_fwd      fwd      = {0};
    struct doca_flow_fwd      miss_fwd = {0};
    struct doca_flow_monitor  mon      = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED
    };
    struct doca_flow_pipe_cfg *pipe_cfg;
    uint16_t rss_queue[1] = {0};   /* RSS verso queue 0 = la GPU RXQ */
    doca_error_t res;

    /* Associa la RXQ alla queue 0 di DOCA Flow per questa porta.
     * Deve essere chiamato dopo doca_ctx_start della RXQ e prima di
     * doca_flow_pipe_create. */
    res = doca_eth_rxq_apply_queue_id(port->rxq_cpu, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_eth_rxq_apply_queue_id: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    /* Forward: RSS a queue 0.
     * outer_flags=0: nessun hashing RSS. Con una sola queue, tutti i
     * pacchetti vanno a queue 0 indipendentemente dall'hashing. */
    fwd.type              = DOCA_FLOW_FWD_RSS;
    fwd.rss_type          = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
    fwd.rss.queues_array  = rss_queue;
    fwd.rss.nr_queues     = 1;
    fwd.rss.outer_flags   = 0;

    /* Miss (pacchetti che non matchano): drop. Con match wildcard non ci
     * sono miss, ma impostiamo DROP per sicurezza. */
    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    res = doca_flow_pipe_cfg_create(&pipe_cfg, port->flow_port);
    if (res != DOCA_SUCCESS) return res;

    doca_flow_pipe_cfg_set_name(pipe_cfg,    "BRIDGE_L2_PIPE");
    doca_flow_pipe_cfg_set_type(pipe_cfg,    DOCA_FLOW_PIPE_BASIC);
    doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);   /* questa pipe è la root */
    /* match template = {0}: wildcard su ethertype, L3 type, L4 type, ecc.
     * = cattura QUALSIASI frame Ethernet (IPv4, ARP, IPv6, MPLS, ...) */
    doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
    doca_flow_pipe_cfg_set_monitor(pipe_cfg, &mon);

    res = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &port->root_pipe);
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_create: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    /* Aggiungi l'unica entry wildcard: match={0} = cattura qualsiasi pacchetto */
    res = doca_flow_pipe_basic_add_entry(
        0,                 /* pipe_queue */
        port->root_pipe,
        &match,            /* match = {0} = any packet */
        0,                 /* flags */
        NULL, NULL, NULL,  /* actions, actions_mask, monitor */
        DOCA_FLOW_ENTRY_FLAGS_NO_WAIT,
        NULL,              /* user context */
        &port->root_entry);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_pipe_basic_add_entry: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    /* Processa le entry in coda per questa porta */
    res = doca_flow_entries_process(port->flow_port, 0, 0, 0);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "[%s] doca_flow_entries_process: %s\n",
                port_label, doca_error_get_descr(res));
        return res;
    }

    printf("[%s] DOCA Flow pronto — BASIC ROOT pipe wildcard → RSS queue 0\n",
           port_label);
    return DOCA_SUCCESS;
}

/* ==========================================================================
 * CLEANUP
 * ==========================================================================
 * Libera le risorse in ordine inverso alla creazione.
 * Il kernel CUDA deve essere già terminato (cudaStreamSynchronize) prima
 * di chiamare questa funzione.
 * Ogni puntatore viene controllato per NULL prima di liberarlo, in modo
 * che la funzione sia sicura anche in caso di errore durante il setup.
 */
static void cleanup_all(struct bridge_port ports[BRIDGE_NUM_PORTS],
                         struct doca_gpu   *gdev,
                         uint64_t          *mac_table_gpu,
                         cudaStream_t       stream,
                         bool               flow_initialized)
{
    int p;

    if (stream)        cudaStreamDestroy(stream);
    if (mac_table_gpu) cudaFree(mac_table_gpu);

    /* Teardown per porta: dal più "esterno" al più "interno" */
    for (p = 0; p < BRIDGE_NUM_PORTS; p++) {
        /* DOCA Flow: prima la pipe, poi la porta */
        if (ports[p].root_pipe) doca_flow_pipe_destroy(ports[p].root_pipe);
        if (ports[p].flow_port) doca_flow_port_stop(ports[p].flow_port);

        /* TXQ */
        if (ports[p].txq_ctx) doca_ctx_stop(ports[p].txq_ctx);
        if (ports[p].txq_cpu) doca_eth_txq_destroy(ports[p].txq_cpu);

        /* RXQ: prima il contesto, poi il buffer GPU, poi la RXQ, poi il mmap */
        if (ports[p].rxq_ctx)  doca_ctx_stop(ports[p].rxq_ctx);
        if (ports[p].rxq_buf && gdev)
            doca_gpu_mem_free(gdev, ports[p].rxq_buf);
        if (ports[p].rxq_cpu)  doca_eth_rxq_destroy(ports[p].rxq_cpu);
        if (ports[p].rxq_mmap) doca_mmap_destroy(ports[p].rxq_mmap);

        if (ports[p].ddev)     doca_dev_close(ports[p].ddev);
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
        "Uso: %s -n <pci_porta0> -n <pci_porta1> -g <pci_gpu> [-i <cuda_id>]\n"
        "\n"
        "  -n  PCI address di una porta della BF2 (specificare due volte)\n"
        "      prima  -n: porta 0 (es. ad:00.0)\n"
        "      seconda -n: porta 1 (es. ad:00.1)\n"
        "  -g  PCI address della GPU A30X (es. b0:00.0)\n"
        "  -i  CUDA device index (default: 0)\n"
        "\n"
        "Esempio:\n"
        "  sudo ip netns exec bf2 ./%s -n ad:00.0 -n ad:00.1 -g b0:00.0\n",
        prog, prog);
}

/* ==========================================================================
 * MAIN
 * ==========================================================================
 */
int main(int argc, char *argv[])
{
    char nic_pci[BRIDGE_NUM_PORTS][DOCA_DEVINFO_PCI_ADDR_SIZE] = {{0}, {0}};
    char gpu_pci[64] = {0};
    int  cuda_id = 0;
    int  nic_count = 0;
    int  opt;

    while ((opt = getopt(argc, argv, "n:g:i:h")) != -1) {
        switch (opt) {
        case 'n':
            if (nic_count >= BRIDGE_NUM_PORTS) {
                fprintf(stderr, "Errore: specificare esattamente 2 flag -n\n");
                return 1;
            }
            strncpy(nic_pci[nic_count++], optarg, DOCA_DEVINFO_PCI_ADDR_SIZE - 1);
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

    if (nic_count < 2 || !gpu_pci[0]) {
        fprintf(stderr, "Errore: servono 2 flag -n e 1 flag -g.\n");
        usage(argv[0]);
        return 1;
    }

    /* ── Variabili principali ─────────────────────────────────────────── */
    doca_error_t          res            = DOCA_SUCCESS;
    struct bridge_port    ports[BRIDGE_NUM_PORTS];
    struct doca_gpu      *gdev           = NULL;
    uint64_t             *mac_table_gpu  = NULL;
    uint32_t             *gpu_exit       = NULL;
    uint32_t             *cpu_exit       = NULL;
    uint64_t             *gpu_fwd        = NULL;
    uint64_t             *cpu_fwd        = NULL;
    cudaStream_t          stream         = NULL;
    bool                  flow_inited    = false;
    const char           *port_labels[BRIDGE_NUM_PORTS] = {"porta 0", "porta 1"};
    struct bridge_kernel_params kp       = {0};

    memset(ports, 0, sizeof(ports));

    cudaSetDevice(cuda_id);

    /* ── 1. Apri entrambe le NIC ──────────────────────────────────────── */
    for (int p = 0; p < BRIDGE_NUM_PORTS; p++) {
        res = open_nic_by_pci(nic_pci[p], &ports[p].ddev);
        if (res != DOCA_SUCCESS) {
            fprintf(stderr, "Apertura NIC %s fallita\n", nic_pci[p]);
            goto cleanup;
        }
        printf("NIC %s aperta (%s)\n", nic_pci[p], port_labels[p]);
    }

    /* ── 2. Init globale DOCA Flow ────────────────────────────────────── */
    res = flow_global_init();
    if (res != DOCA_SUCCESS) goto cleanup;
    flow_inited = true;

    /* ── 3. Avvia le porte DOCA Flow (ID univoci: 0 e 1) ──────────────── */
    for (int p = 0; p < BRIDGE_NUM_PORTS; p++) {
        res = flow_start_port(&ports[p], p);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 4. Crea il contesto GPU (A30X) ──────────────────────────────── */
    res = doca_gpu_create(gpu_pci, &gdev);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "doca_gpu_create (%s): %s\n", gpu_pci,
                doca_error_get_descr(res));
        goto cleanup;
    }
    printf("GPU %s (CUDA device %d) aperta\n", gpu_pci, cuda_id);

    /* ── 5. Setup RXQ per entrambe le porte ──────────────────────────────
     * Nota: porta 0 viene registrata con ddev1 (e viceversa) per cross-port.
     * Questo è possibile solo perché abbiamo aperto entrambi i ddev al passo 1.
     */
    res = setup_port_rxq(&ports[0], ports[1].ddev, gdev, cuda_id, port_labels[0]);
    if (res != DOCA_SUCCESS) goto cleanup;

    res = setup_port_rxq(&ports[1], ports[0].ddev, gdev, cuda_id, port_labels[1]);
    if (res != DOCA_SUCCESS) goto cleanup;

    /* ── 6. Setup TXQ per entrambe le porte ─────────────────────────── */
    for (int p = 0; p < BRIDGE_NUM_PORTS; p++) {
        res = setup_port_txq(&ports[p], gdev, port_labels[p]);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 7. Setup DOCA Flow pipe per entrambe le porte ──────────────── */
    for (int p = 0; p < BRIDGE_NUM_PORTS; p++) {
        res = setup_port_flow(&ports[p], port_labels[p]);
        if (res != DOCA_SUCCESS) goto cleanup;
    }

    /* ── 8. Alloca MAC table in GPU memory ───────────────────────────── */
    if (cudaMalloc(&mac_table_gpu,
                   (size_t)MAC_TABLE_SIZE * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc MAC table fallita\n");
        goto cleanup;
    }
    /* Inizializza a zero: tutti gli slot sono vuoti (valid bit = 0) */
    cudaMemset(mac_table_gpu, 0, (size_t)MAC_TABLE_SIZE * sizeof(uint64_t));

    /* ── 9. Alloca exit_cond (GPU_CPU) ───────────────────────────────────
     * GPU_CPU: la memoria è in GPU (accesso veloce per polling del kernel)
     * ma accessibile anche dal CPU (per scrivere 1 al Ctrl+C).
     * Il kernel fa polling su gpu_exit; il CPU scrive su cpu_exit.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint32_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void **)&gpu_exit, (void **)&cpu_exit);
    if (res != DOCA_SUCCESS || !gpu_exit) {
        fprintf(stderr, "alloc exit_cond: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_exit = 0;   /* 0 = il kernel continua */

    /* ── 10. Alloca fwd_count (CPU_GPU) ──────────────────────────────────
     * CPU_GPU: la memoria è nel lato CPU (accesso veloce per lettura del CPU)
     * ma accessibile dalla GPU (il kernel CUDA ci scrive).
     * Il kernel scrive su gpu_fwd; il CPU legge su cpu_fwd dopo sync.
     */
    res = doca_gpu_mem_alloc(gdev, sizeof(uint64_t), system_page_size(),
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void **)&gpu_fwd, (void **)&cpu_fwd);
    if (res != DOCA_SUCCESS || !gpu_fwd) {
        fprintf(stderr, "alloc fwd_count: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }
    *cpu_fwd = 0;

    /* ── 11. Crea CUDA stream ─────────────────────────────────────────── */
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags fallita\n");
        goto cleanup;
    }

    /* ── 12. Prepara parametri kernel ─────────────────────────────────── */
    kp.rxq_gpu[0]          = ports[0].rxq_gpu;
    kp.rxq_gpu[1]          = ports[1].rxq_gpu;
    kp.txq_gpu[0]          = ports[0].txq_gpu;
    kp.txq_gpu[1]          = ports[1].txq_gpu;
    /* mkey[0]: mkey del buffer di porta 0, valido per la NIC di porta 1.
     * Usato nei WQE di txq1 per leggere il buffer GPU di rxq0 (p0→p1). */
    kp.rxq_mkey_for_other[0] = ports[0].rxq_mkey_for_other_txq;
    /* mkey[1]: mkey del buffer di porta 1, valido per la NIC di porta 0.
     * Usato nei WQE di txq0 per leggere il buffer GPU di rxq1 (p1→p0). */
    kp.rxq_mkey_for_other[1] = ports[1].rxq_mkey_for_other_txq;
    kp.mac_table           = mac_table_gpu;
    kp.exit_cond           = gpu_exit;
    kp.fwd_count           = gpu_fwd;

    /* ── 13. Lancia il kernel CUDA ───────────────────────────────────── */
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\nGPU Bridge avviato:\n");
    printf("  Porta 0: %s\n", nic_pci[0]);
    printf("  Porta 1: %s\n", nic_pci[1]);
    printf("  GPU:     %s (CUDA device %d)\n", gpu_pci, cuda_id);
    printf("  MAC table: %d slot (FNV-1a hash, linear probing)\n", MAC_TABLE_SIZE);
    printf("Premi Ctrl+C per fermare.\n\n");

    res = kernel_launch_bridge(stream, &kp);
    if (res != DOCA_SUCCESS) {
        fprintf(stderr, "kernel_launch_bridge: %s\n", doca_error_get_descr(res));
        goto cleanup;
    }

    /* ── 14. Loop CPU: aspetta Ctrl+C ────────────────────────────────── */
    while (!DOCA_GPUNETIO_VOLATILE(g_force_quit))
        ;   /* busy wait sulla variabile volatile: Ctrl+C setta g_force_quit */

    /* ── 15. Ferma il kernel ─────────────────────────────────────────── */
    DOCA_GPUNETIO_VOLATILE(*cpu_exit) = 1;   /* segnala al kernel di uscire */
    cudaStreamSynchronize(stream);            /* aspetta che il kernel finisca */

    printf("\nFermato. Totale pacchetti forwardati: %lu\n", *cpu_fwd);

cleanup:
    /* Libera le variabili di sincronizzazione prima del cleanup generale */
    if (gpu_exit && gdev) { doca_gpu_mem_free(gdev, gpu_exit); gpu_exit = NULL; }
    if (gpu_fwd  && gdev) { doca_gpu_mem_free(gdev, gpu_fwd);  gpu_fwd  = NULL; }

    cleanup_all(ports, gdev, mac_table_gpu, stream, flow_inited);
    printf("Done.\n");
    return (res == DOCA_SUCCESS) ? 0 : 1;
}
