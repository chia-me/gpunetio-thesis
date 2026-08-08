/*
 * gpu_router_kernel.cu — kernel CUDA per il GPU L3 Router a N porte.
 *
 * ARCHITETTURA DEL KERNEL (identica nell'impalcatura al bridge — vedi
 * prog_gpu_bridge/gpu_bridge_kernel.cu per il confronto diretto — ma
 * diversa nella logica di decisione e nella manipolazione del pacchetto):
 *
 *   Un solo kernel persistente, 1 blocco, 32 thread (= 1 warp NVIDIA).
 *   Il kernel gira per sempre finché la CPU non scrive 1 in exit_cond.
 *
 *   Il loop principale itera su tutte le N porte sorgente:
 *
 *     per ogni porta src (0..n_ports-1):
 *       Riceve da rxq[src], processa, invia verso UNA porta di uscita
 *       (o scarta).
 *
 *   Per ogni porta sorgente, il processing ha tre fasi:
 *
 *     Fase 1 — ricezione (tutti i 32 thread cooperano, EXEC_SCOPE_BLOCK).
 *
 *     Fase 2 — parallela (tutti i 32 thread contemporaneamente):
 *       Ogni thread gestisce i propri pacchetti (t, t+32, t+64, ...):
 *         1. Valida l'header IPv4 (versione, IHL) — difesa in profondità,
 *            anche se DOCA Flow ha già filtrato per EtherType IPv4 in HW.
 *         2. TTL check: se il pacchetto non sopravviverebbe al decremento
 *            (ttl <= 1), viene scartato (nessun ICMP Time Exceeded: fuori
 *            scope, vedi README).
 *         3. FIB lookup (Longest Prefix Match) sull'IP di destinazione.
 *            Nessun match -> scarta (nessuna default route implicita).
 *         4. Se si inoltra: decrementa la TTL, ricalcola il checksum IPv4,
 *            riscrive MAC sorgente (= interfaccia di uscita) e MAC
 *            destinazione (= next-hop della rotta). Tutto IN PLACE nel
 *            buffer GPU di ricezione: zero copie.
 *         5. atomicAdd su next_wqe_slot[eg_port] per ottenere uno slot
 *            libero e consecutivo nel ring della TXQ di uscita, poi riempie
 *            il WQE (zero-copy cross-port, come nel bridge).
 *
 *     Fase 3 — seriale (solo thread 0): fixup NOTIFY + submit + poll,
 *       identico meccanismo del bridge (vedi il commento esteso lì).
 *
 * DIFFERENZE CHIAVE RISPETTO AL BRIDGE (perché qui non serve una bitmask
 * "egress_mask" a più bit):
 *
 *   Un bridge L2 può FLOODARE (un frame va potenzialmente a N-1 porte
 *   quando il MAC destinazione è sconosciuto). Un router L3 che instrada
 *   traffico unicast IPv4 non floода mai: ogni pacchetto ha ESATTAMENTE
 *   una rotta che lo riguarda (o nessuna, e allora si scarta). Quindi la
 *   "egress mask" del bridge si riduce qui a un singolo intero eg_port
 *   (o -1 per "scarta"), con un unico bit nel caso venga espresso come
 *   mask. Il meccanismo di riempimento del WQE (atomicAdd per lo slot,
 *   atomicMax per identificare l'ultimo WQE del batch da marcare NOTIFY)
 *   resta lo stesso del bridge perché più pacchetti nello stesso batch,
 *   pur non "floodando" mai singolarmente, possono comunque avere eg_port
 *   diversi tra loro (è un router a N porte, non un semplice repeater):
 *   la TXQ di ogni porta riceve comunque un flusso di WQE da consolidare.
 *
 * PERCHÉ LA FIB NON HA BISOGNO DI ATOMICI (a differenza della mac_table
 * del bridge): è caricata UNA VOLTA dalla CPU prima del lancio del kernel
 * (routes.txt -> RIB ordinata -> cudaMemcpy) e non viene mai più scritta.
 * Nessun MAC learning qui: le rotte sono statiche per tutta la vita del
 * processo. È una lettura pura, quindi nessuna sincronizzazione richiesta.
 */

#include <stdio.h>

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gpu_router.h"

/* =========================================================================
 * ENUM — ESITO DELLA DECISIONE DI ROUTING PER UN SINGOLO PACCHETTO
 * ========================================================================= */
enum router_verdict {
    RTR_FORWARD = 0,
    RTR_DROP_MALFORMED,     /* IPv4 con version/IHL non validi */
    RTR_DROP_TTL_EXPIRED,   /* TTL <= 1: non sopravviverebbe al decremento */
    RTR_DROP_NO_ROUTE,      /* nessuna rotta della FIB copre questo dst_addr */
};

/* =========================================================================
 * FUNZIONI DEVICE — CHECKSUM IPv4
 * =========================================================================
 * Algoritmo standard RFC 791 §3.1 / RFC 1071: complemento a uno della
 * somma (in complemento a uno) di tutte le "parole" a 16 bit dell'header,
 * con il campo checksum stesso azzerato durante il calcolo.
 *
 * Qui viene ricalcolato per INTERO ad ogni pacchetto instradato (fino a
 * 60 byte / 30 parole a 16 bit, per un header con opzioni IP al massimo).
 * Un'ottimizzazione classica (RFC 1624) aggiornerebbe il checksum in modo
 * incrementale sapendo che cambia solo il byte TTL — risparmiando quasi
 * tutte le iterazioni del loop — ma la ricomputazione completa è scelta
 * qui deliberatamente: è di gran lunga più facile da leggere, verificare
 * e fidarsi (nessun rischio di un bug sottile nella formula incrementale),
 * e il costo (<=30 somme a 16 bit per pacchetto) è trascurabile per un
 * kernel già dominato dal costo della ricezione/trasmissione via NIC.
 */
static __device__ __forceinline__ uint16_t ipv4_header_checksum(const uint16_t *hdr16, int hdr_len_bytes)
{
    uint32_t sum = 0;
    int n_words = hdr_len_bytes >> 1;

    for (int i = 0; i < n_words; i++)
        sum += hdr16[i];

    /* Ripiega i riporti oltre il bit 16 finché non ne restano (al più
     * due iterazioni sono sempre sufficienti per un header IPv4). */
    while (sum >> 16)
        sum = (sum & 0xFFFFu) + (sum >> 16);

    return (uint16_t)(~sum);
}

/* =========================================================================
 * FUNZIONE DEVICE — FIB LOOKUP (Longest Prefix Match)
 * =========================================================================
 * ALGORITMO: scansione lineare di un array ORDINATO per lunghezza di
 * prefisso decrescente (l'ordinamento è fatto una tantum sulla CPU, vedi
 * route_cmp_by_prefixlen_desc in gpu_router.c). Il PRIMO elemento che fa
 * match (dst_addr & mask == network) è per costruzione quello con il
 * prefisso più lungo possibile tra tutti quelli che fanno match: è
 * esattamente la definizione di Longest Prefix Match.
 *
 * PERCHÉ NON UNA TRIE (Patricia trie / DIR-24-8, usate dai router HW
 * reali con centinaia di migliaia di rotte BGP): con un numero di rotte
 * dell'ordine delle decine/centinaia (MAX_ROUTES = 1024), una scansione
 * lineare di al più 1024 confronti a 32 bit è già ben al di sotto del
 * microsecondo per thread, e il codice è enormemente più semplice da
 * leggere e verificare rispetto a una struttura ad albero con bit-walk.
 * Per un router con una tabella BGP completa (~1M rotte) questa scelta
 * andrebbe rivista: è un compromesso esplicito, non un limite tecnico
 * della GPU.
 *
 * dst_addr_be: indirizzo IP di destinazione COSÌ COME appare nel pacchetto
 * (network byte order, nessuno swap fatto né necessario — vedi il
 * commento su route_entry in gpu_router.h).
 */
static __device__ __forceinline__ bool fib_lookup(const struct route_entry *fib,
                                                    uint32_t fib_size,
                                                    uint32_t dst_addr_be,
                                                    int *out_egress_port,
                                                    const uint8_t **out_next_hop_mac)
{
    for (uint32_t i = 0; i < fib_size; i++) {
        if ((dst_addr_be & fib[i].mask) == fib[i].network) {
            *out_egress_port  = fib[i].egress_port;
            *out_next_hop_mac = fib[i].next_hop_mac;
            return true;
        }
    }
    return false;
}

/* =========================================================================
 * FUNZIONE DEVICE — DECISIONE + RISCRITTURA DI UN SINGOLO PACCHETTO
 * =========================================================================
 * Esegue tutto il lavoro "L3" su un pacchetto IPv4 già confermato tale da
 * DOCA Flow (EtherType 0x0800 nel match della pipe — vedi setup_port_flow
 * in gpu_router.c). Ritorna il verdetto e, se RTR_FORWARD, valorizza
 * *out_egress_port con la porta su cui il chiamante deve accodare il WQE.
 *
 * EFFETTI COLLATERALI (solo se ritorna RTR_FORWARD):
 *   Il buffer del pacchetto (in memoria GPU, quello della RXQ sorgente)
 *   viene MODIFICATO IN PLACE:
 *     - TTL decrementata di 1
 *     - checksum IPv4 ricalcolato
 *     - MAC sorgente sovrascritto con own_mac (l'interfaccia di uscita)
 *     - MAC destinazione sovrascritto con il next-hop della rotta
 *   Questo è sicuro perché ogni pacchetto è processato da un solo thread
 *   (nessuna scrittura concorrente sullo stesso buffer) e la modifica
 *   avviene PRIMA che il WQE (preparato subito dopo da chi chiama questa
 *   funzione) venga sottomesso alla NIC per la trasmissione.
 */
static __device__ enum router_verdict route_and_rewrite_packet(
        uint64_t pkt_addr,
        const struct route_entry *fib,
        uint32_t fib_size,
        const uint8_t own_mac_of_egress_port[MAX_N_PORTS][ETHER_ADDR_LEN],
        int *out_egress_port)
{
    struct eth_ip_hdr *hdr = (struct eth_ip_hdr *)(uintptr_t)pkt_addr;
    struct ipv4_hdr *ip = &hdr->l3_hdr;

    /* ── Validazione IPv4 (difesa in profondità) ──────────────────────
     * DOCA Flow ha già garantito EtherType == 0x0800, ma non ha guardato
     * DENTRO il pacchetto: un mittente malevolo o bacato potrebbe mandare
     * un frame con EtherType IPv4 ma un header IP corrotto. version deve
     * essere 4; ihl (in parole da 32 bit) deve essere almeno 5 (20 byte,
     * l'header minimo senza opzioni), altrimenti i campi che leggiamo
     * sotto (checksum, ttl, dst_addr, ...) potrebbero non esistere nel
     * pacchetto reale o il conteggio byte del checksum sarebbe sbagliato. */
    if (ip->version != 4 || ip->ihl < 5)
        return RTR_DROP_MALFORMED;

    /* ── TTL check ──────────────────────────────────────────────────
     * Un router IP DEVE scartare un pacchetto la cui TTL, decrementata,
     * arriverebbe a 0 (RFC 791): il pacchetto ha esaurito il numero
     * massimo di hop consentiti. Un router "completo" risponderebbe con
     * un ICMP Time Exceeded al mittente; qui ci limitiamo a scartare
     * silenziosamente (implementare la generazione di un pacchetto ICMP
     * è un'estensione naturale ma fuori dallo scope di questo esempio
     * didattico — vedi README). */
    if (ip->time_to_live <= 1)
        return RTR_DROP_TTL_EXPIRED;

    /* ── FIB lookup (Longest Prefix Match) ─────────────────────────── */
    int egress_port;
    const uint8_t *next_hop_mac;
    if (!fib_lookup(fib, fib_size, ip->dst_addr, &egress_port, &next_hop_mac))
        return RTR_DROP_NO_ROUTE;

    /* ── Riscrittura header (IN PLACE) ──────────────────────────────
     * NOTA: instradare un pacchetto sulla stessa porta da cui è arrivato
     * (egress_port == src) è permesso: può essere una configurazione
     * legittima (es. due sotto-reti raggiungibili dalla stessa interfaccia
     * fisica). Un router "completo" invierebbe qui un ICMP Redirect al
     * mittente per suggerirgli un percorso più diretto; non implementato
     * (stesso discorso del TTL: fuori scope, vedi README). */

    ip->time_to_live -= 1;

    /* Azzera il checksum PRIMA di ricalcolarlo: è lui stesso una delle
     * "parole" sommate dall'algoritmo, deve valere 0 durante il calcolo. */
    ip->hdr_checksum = 0;
    ip->hdr_checksum = ipv4_header_checksum((const uint16_t *)ip, (int)ip->ihl * 4);

    /* MAC sorgente = interfaccia di uscita; MAC destinazione = next-hop
     * della rotta. L'EtherType (0x0800) resta invariato: il payload è
     * sempre IPv4. */
#pragma unroll
    for (int i = 0; i < ETHER_ADDR_LEN; i++) {
        hdr->l2_hdr.s_addr_bytes[i] = own_mac_of_egress_port[egress_port][i];
        hdr->l2_hdr.d_addr_bytes[i] = next_hop_mac[i];
    }

    *out_egress_port = egress_port;
    return RTR_FORWARD;
}

/* =========================================================================
 * KERNEL PRINCIPALE — router_kernel
 * =========================================================================
 * Dimensioni: <<<1, 32, 0, stream>>> (1 blocco, 32 thread = 1 warp).
 * Vedi gpu_bridge_kernel.cu per la spiegazione completa (identica qui) del
 * perché EXEC_SCOPE_BLOCK richiede tutti i 32 thread del warp per la
 * ricezione, e del meccanismo atomicAdd/atomicMax per il riempimento
 * lock-free dei WQE nel ring circolare della TXQ.
 */
__global__ void router_kernel(struct router_kernel_params kp)
{
    /* ── Shared memory ─────────────────────────────────────────────────── */
    __shared__ uint64_t rx_first_pkt_idx;
    __shared__ uint32_t rx_pkt_count;
    __shared__ struct doca_gpu_dev_eth_rxq_attr rx_attr[MAX_RX_NUM_PKTS];

    /* next_wqe_slot[q] / last_wqe_key[q]: vedi gpu_bridge_kernel.cu per la
     * spiegazione completa del meccanismo (identico qui). */
    __shared__ uint32_t next_wqe_slot[MAX_N_PORTS];
    __shared__ uint64_t last_wqe_key[MAX_N_PORTS];

    /* Contatori diagnostici di batch, consolidati da thread 0 a fine batch
     * nei totali persistenti (drop_no_route_total, ecc.). */
    __shared__ uint32_t batch_no_route;
    __shared__ uint32_t batch_ttl_expired;
    __shared__ uint32_t batch_malformed;

    /* wqe_base[q]: indice assoluto del prossimo WQE da scrivere per TXQ q.
     * Deve essere shared: letto da tutti i thread in Fase 2 (vedi bridge). */
    __shared__ uint64_t wqe_base[MAX_N_PORTS];

    uint64_t cqe_idx[MAX_N_PORTS];
    uint64_t tot_fwd = 0;
    uint64_t rx_pkt_total_local[MAX_N_PORTS];
    uint64_t no_route_total = 0, ttl_expired_total = 0, malformed_total = 0;

    doca_error_t ret;

    if (threadIdx.x == 0) {
        for (int q = 0; q < MAX_N_PORTS; q++) {
            wqe_base[q] = 0;
            cqe_idx[q]  = 0;
        }
        for (int q = 0; q < MAX_N_PORTS; q++)
            rx_pkt_total_local[q] = 0;
    }
    __syncthreads();

    /* =====================================================================
     * LOOP PRINCIPALE
     * ===================================================================== */
    while (DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) == 0) {

        for (int src = 0;
             src < kp.n_ports && DOCA_GPUNETIO_VOLATILE(*kp.exit_cond) == 0;
             src++)
        {
            /* ══════════════════════════════════════════════════════════
             * FASE 1: RICEZIONE (tutti i 32 thread cooperano)
             * ══════════════════════════════════════════════════════════ */
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

            /* ══════════════════════════════════════════════════════════
             * FASE 2A: RESET CONTATORI BATCH (solo thread 0)
             * ══════════════════════════════════════════════════════════ */
            if (threadIdx.x == 0) {
                for (int q = 0; q < kp.n_ports; q++) {
                    next_wqe_slot[q] = 0;
                    last_wqe_key[q]  = 0;
                }
                batch_no_route    = 0;
                batch_ttl_expired = 0;
                batch_malformed   = 0;
                rx_pkt_total_local[src] += rx_pkt_count;
            }
            __syncthreads();

            /* ══════════════════════════════════════════════════════════
             * FASE 2B: ROUTING + RISCRITTURA + WQE FILL PARALLELO
             *
             * Tutti i 32 thread in parallelo, thread t gestisce i
             * pacchetti t, t+32, t+64, ... (come nel bridge).
             *
             * Per ogni pacchetto:
             *   1. route_and_rewrite_packet(): valida, decide (FIB LPM),
             *      se forwardabile riscrive TTL/checksum/MAC IN PLACE.
             *   2. Se RTR_FORWARD: un solo bit di egress (mai flooding —
             *      vedi il commento in testa al file), quindi un solo WQE:
             *      atomicAdd su next_wqe_slot[eg_port] per lo slot,
             *      atomicMax su last_wqe_key[eg_port] per il fixup NOTIFY,
             *      riempi il WQE zero-copy cross-port (mkey da
             *      rxq_mkey_cross[src][eg_port], come nel bridge).
             *   3. Se scartato: incrementa il contatore diagnostico
             *      corrispondente. Nessun WQE, nessuna trasmissione.
             * ══════════════════════════════════════════════════════════ */
            uint32_t pkt_idx = threadIdx.x;
            while (pkt_idx < rx_pkt_count) {

                uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                    kp.rxq_gpu[src], rx_first_pkt_idx + pkt_idx);

                int eg_port = -1;
                enum router_verdict verdict = route_and_rewrite_packet(
                    pkt_addr, kp.fib, kp.fib_size, kp.port_mac, &eg_port);

                switch (verdict) {
                case RTR_DROP_MALFORMED:
                    atomicAdd(&batch_malformed, 1u);
                    break;
                case RTR_DROP_TTL_EXPIRED:
                    atomicAdd(&batch_ttl_expired, 1u);
                    break;
                case RTR_DROP_NO_ROUTE:
                    atomicAdd(&batch_no_route, 1u);
                    break;
                case RTR_FORWARD: {
                    uint32_t mkey = kp.rxq_mkey_cross[src][eg_port];
                    uint32_t slot = atomicAdd(&next_wqe_slot[eg_port], 1u);

                    uint64_t key = ((uint64_t)slot << 32) | (uint64_t)pkt_idx;
                    atomicMax((unsigned long long *)&last_wqe_key[eg_port],
                              (unsigned long long)key);

                    struct doca_gpu_dev_eth_txq_wqe *wqe =
                        doca_gpu_dev_eth_txq_get_wqe_ptr(
                            kp.txq_gpu[eg_port], wqe_base[eg_port] + slot);
                    doca_gpu_dev_eth_txq_wqe_prepare_send(
                        kp.txq_gpu[eg_port], wqe, wqe_base[eg_port] + slot,
                        pkt_addr, mkey, rx_attr[pkt_idx].bytes,
                        DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);
                    break;
                }
                }

                pkt_idx += blockDim.x;
            }

            __syncthreads();

            /* ══════════════════════════════════════════════════════════
             * FASE 3: FIXUP NOTIFY + SUBMIT + POLL (solo thread 0)
             * Meccanismo identico al bridge — vedi gpu_bridge_kernel.cu.
             * ══════════════════════════════════════════════════════════ */
            if (threadIdx.x == 0) {
                for (int q = 0; q < kp.n_ports; q++) {
                    uint32_t n = next_wqe_slot[q];
                    if (n == 0)
                        continue;

                    uint32_t last_pkt_idx = (uint32_t)(last_wqe_key[q] & 0xFFFFFFFFu);
                    uint64_t last_pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                        kp.rxq_gpu[src], rx_first_pkt_idx + last_pkt_idx);
                    uint32_t last_pkt_bytes = rx_attr[last_pkt_idx].bytes;
                    uint32_t last_pkt_mkey  = kp.rxq_mkey_cross[src][q];

                    uint64_t last_slot = wqe_base[q] + n - 1;
                    struct doca_gpu_dev_eth_txq_wqe *last_wqe =
                        doca_gpu_dev_eth_txq_get_wqe_ptr(kp.txq_gpu[q], last_slot);
                    doca_gpu_dev_eth_txq_wqe_prepare_send(
                        kp.txq_gpu[q], last_wqe, last_slot,
                        last_pkt_addr, last_pkt_mkey, last_pkt_bytes,
                        DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY);

                    doca_gpu_dev_eth_txq_submit(kp.txq_gpu[q], wqe_base[q] + n);

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

                no_route_total    += batch_no_route;
                ttl_expired_total += batch_ttl_expired;
                malformed_total   += batch_malformed;
            }
            __syncthreads();

        } /* for src */

        /* Pubblica i contatori diagnostici verso la CPU una volta per ogni
         * giro completo su tutte le porte (stesso schema del bridge:
         * __threadfence_system rende visibili le scritture precedenti). */
        if (threadIdx.x == 0) {
            for (int q = 0; q < kp.n_ports; q++)
                kp.rx_pkt_total[q] = rx_pkt_total_local[q];
            *kp.drop_no_route    = no_route_total;
            *kp.drop_ttl_expired = ttl_expired_total;
            *kp.drop_malformed   = malformed_total;
            *kp.fwd_count        = tot_fwd;
            __threadfence_system();
        }
        __syncthreads();

    } /* while (!exit_cond) */

    /* Pubblicazione finale, ridondante ma garantisce che l'ultimo batch
     * prima di exit_cond sia sempre riflesso nel report finale. */
    if (threadIdx.x == 0) {
        for (int q = 0; q < kp.n_ports; q++)
            kp.rx_pkt_total[q] = rx_pkt_total_local[q];
        *kp.drop_no_route    = no_route_total;
        *kp.drop_ttl_expired = ttl_expired_total;
        *kp.drop_malformed   = malformed_total;
        *kp.fwd_count        = tot_fwd;
        __threadfence_system();
    }
}

/* =========================================================================
 * WRAPPER EXTERN "C"
 * ========================================================================= */
extern "C" {

doca_error_t kernel_launch_router(cudaStream_t stream,
                                   struct router_kernel_params *kp)
{
    cudaError_t cuda_ret;

    if (!kp || !kp->fib || kp->fib_size == 0 || !kp->exit_cond || !kp->fwd_count)
        return DOCA_ERROR_INVALID_VALUE;

    if (!kp->rx_pkt_total || !kp->drop_no_route || !kp->drop_ttl_expired || !kp->drop_malformed)
        return DOCA_ERROR_INVALID_VALUE;

    if (kp->n_ports < 2 || kp->n_ports > MAX_N_PORTS)
        return DOCA_ERROR_INVALID_VALUE;

    for (int i = 0; i < kp->n_ports; i++) {
        if (!kp->rxq_gpu[i] || !kp->txq_gpu[i])
            return DOCA_ERROR_INVALID_VALUE;
    }

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA error before kernel launch: %s\n", cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    router_kernel<<<1, ROUTER_BLOCK_THREADS, 0, stream>>>(*kp);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern "C" */
