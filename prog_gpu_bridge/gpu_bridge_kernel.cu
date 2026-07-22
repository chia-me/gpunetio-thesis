/*
 * gpu_bridge_kernel.cu — kernel CUDA per il GPU L2 Bridge.
 *
 * ARCHITETTURA DEL KERNEL:
 *
 *   Un solo kernel persistente, 1 blocco, 32 thread (= 1 warp NVIDIA).
 *   Il kernel gira per sempre finché il CPU non scrive 1 in exit_cond.
 *
 *   Il loop principale alterna tra due direzioni:
 *
 *     Direzione A: porta 0 → porta 1
 *       Riceve da rxq0, processa, invia su txq1.
 *
 *     Direzione B: porta 1 → porta 0
 *       Riceve da rxq1, processa, invia su txq0.
 *
 *   Per ogni direzione, la fase di elaborazione ha due sotto-fasi:
 *
 *     Fase parallela (tutti e 32 i thread in contemporanea):
 *       Ogni thread elabora i propri pacchetti (indici t, t+32, t+64, ...):
 *         1. Legge dst_mac e src_mac dall'header Ethernet
 *         2. Backward learning: inserisce src_mac → porta_ingresso nella FIB
 *         3. FIB lookup: cerca dst_mac, ottiene porta_uscita o -1 (flood)
 *         4. Scrive fwd_decision[i] = 1 (forward) o 0 (drop)
 *
 *     Fase seriale (solo thread 0):
 *       Legge tutti i fwd_decision[], riempie WQE consecutivi sulla TXQ
 *       di destinazione, chiama submit (doorbell), aspetta CQE.
 *       Thread 0 deve essere seriale per assegnare indici WQE contigui
 *       (non possiamo avere buchi nella sequenza di WQE).
 *
 * ZERO-COPY CROSS-PORT:
 *   I WQE di txq1 puntano direttamente ai dati in GPU memory di rxq0.
 *   La NIC porta 1 fa DMA direttamente dalla GPU senza nessuna copia.
 *   Il mkey rxq0_mkey_for_txq1 è necessario perché la NIC porta 1
 *   ha un Protection Domain RDMA diverso dalla NIC porta 0:
 *   deve avere una chiave valida per accedere alla memoria GPU di porta 0.
 */

#include <stdio.h>

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gpu_bridge.h"

/* =========================================================================
 * FUNZIONI DEVICE — MAC TABLE
 * =========================================================================
 * Funzioni __device__: girano solo sulla GPU, chiamate dal kernel.
 * __forceinline__: il compilatore le inline (elimina overhead chiamata).
 */

/*
 * mac_to_u64: converte 6 byte di MAC address (network byte order) in uint64_t.
 *
 * Header Ethernet: il byte [0] è il più significativo (MAC big-endian).
 * Esempio: MAC aa:bb:cc:dd:ee:ff → uint64_t 0x0000aabbccddeeff
 *
 * Perché uint64_t e non array di byte?
 * Perché è più comodo fare confronti con == e usare come chiave hash.
 * I 16 bit superiori sono sempre 0 (MAC è 48 bit, uint64_t è 64 bit).
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
 * mac_hash: hash FNV-1a (Fowler–Noll–Vo) di un MAC a 48 bit.
 *
 * FNV-1a: per ogni byte, XOR con l'hash corrente, poi moltiplica per una costante.
 * È veloce (no divisione, no branch), ha buona distribuzione su input corti.
 *
 * Il risultato viene mascherato con (MAC_TABLE_SIZE - 1) = 4095.
 * Poiché MAC_TABLE_SIZE è potenza di 2, (h & 4095) == (h % 4096) ma senza
 * divisione hardware (molto più veloce su GPU dove la divisione è costosa).
 */
static __device__ __forceinline__ uint32_t mac_hash(uint64_t mac48)
{
    uint32_t h = 2166136261u;          /* FNV offset basis 32-bit */
    h ^= (uint8_t)(mac48 >> 40); h *= 16777619u;   /* FNV prime 32-bit */
    h ^= (uint8_t)(mac48 >> 32); h *= 16777619u;
    h ^= (uint8_t)(mac48 >> 24); h *= 16777619u;
    h ^= (uint8_t)(mac48 >> 16); h *= 16777619u;
    h ^= (uint8_t)(mac48 >>  8); h *= 16777619u;
    h ^= (uint8_t)(mac48);       h *= 16777619u;
    return h & (MAC_TABLE_SIZE - 1);
}

/*
 * mac_learn: registra nella FIB che 'mac48' è raggiungibile sulla 'porta'.
 *
 * FORMATO ENTRY uint64_t:
 *   bit 63:   valid (1 = slot occupato, 0 = slot vuoto)
 *   bit 48:   numero porta (0 o 1)
 *   bit 0-47: MAC address
 *
 * ALGORITMO — linear probing con atomici per thread-safety:
 *
 *   slot = mac_hash(mac48)
 *   per probe = 0 .. MAC_TABLE_PROBE_MAX:
 *     leggi mac_table[slot] (volatile)
 *     se slot vuoto (val == 0):
 *       atomicCAS(slot, 0, nuova_entry)
 *       se CAS riuscito (old == 0): return (inserimento ok)
 *       se CAS fallito: un altro thread ha preso il posto → rileggi e riprova
 *     se slot ha questo stesso MAC:
 *       atomicExch(slot, nuova_entry) → aggiorna porta
 *       return
 *     altrimenti: slot occupato da MAC diverso → linear probe al prossimo
 *
 * THREAD-SAFETY:
 *   atomicCAS e atomicExch operano atomicamente su uint64_t in global memory.
 *   La consistenza è eventual: se due thread imparano lo stesso MAC
 *   contemporaneamente, uno dei due vince il CAS; l'altro rilegge e trova
 *   il proprio MAC già presente, quindi fa atomicExch per aggiornare la porta.
 *   Risultato: sempre consistente, mai corrotto.
 *
 * CASO TABELLA PIENA:
 *   Se non troviamo uno slot libero entro MAC_TABLE_PROBE_MAX tentativi,
 *   rinunciamo silenziosamente. Il pacchetto sarà trattato come "flooding"
 *   (destinazione sconosciuta), comportamento corretto per un bridge.
 */
static __device__ void mac_learn(uint64_t *mac_table, uint64_t mac48, int porta)
{
    uint64_t nuova_entry = MAC_ENTRY_VALID_BIT |
                           ((uint64_t)(uint32_t)porta << MAC_ENTRY_PORT_SHIFT) |
                           mac48;
    uint32_t slot = mac_hash(mac48);

    for (int probe = 0; probe < MAC_TABLE_PROBE_MAX; probe++) {
        /* Lettura volatile: impedisce al compilatore di cacheare in un registro */
        uint64_t val = *(volatile uint64_t *)&mac_table[slot];

        if (val == 0) {
            /* Slot vuoto: prova ad occuparlo */
            uint64_t old = atomicCAS((unsigned long long *)&mac_table[slot],
                                     0ULL,
                                     (unsigned long long)nuova_entry);
            if (old == 0)
                return;  /* CAS riuscito: inserimento completato */

            /* CAS fallito: un altro thread ha riempito il posto.
             * 'old' è il valore che c'era quando abbiamo tentato.
             * Controlliamo se è il nostro stesso MAC (caso aggiornamento). */
            val = old;
        }

        /* Slot occupato: è il nostro MAC? */
        if ((val & MAC_48BIT_MASK) == mac48) {
            /* Stesso MAC già presente: aggiorna la porta */
            atomicExch((unsigned long long *)&mac_table[slot],
                       (unsigned long long)nuova_entry);
            return;
        }

        /* MAC diverso: linear probe al prossimo slot */
        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
    /* Tabella troppo piena per questo bucket: rinuncia */
}

/*
 * mac_lookup: cerca 'mac48' nella FIB.
 *
 * Ritorna:
 *    0 o 1 = porta dove si trova il MAC (trovato)
 *   -1     = MAC non trovato (flood: invia all'altra porta)
 *
 * Il linear probing si ferma al primo slot VUOTO (entry == 0).
 * Invariante del linear probing: se il MAC esistesse, sarebbe stato
 * trovato PRIMA di un slot vuoto (non ci sono "buchi" nelle catene
 * di probe, perché non cancelliamo mai entry dalla tabella).
 *
 * Lettura volatile: necessaria perché altri thread scrivono mac_table
 * con atomici. Senza volatile il compilatore potrebbe cacheare il valore
 * e non "vedere" le scritture concorrenti.
 */
static __device__ int mac_lookup(const uint64_t *mac_table, uint64_t mac48)
{
    uint32_t slot = mac_hash(mac48);

    for (int probe = 0; probe < MAC_TABLE_PROBE_MAX; probe++) {
        uint64_t entry = *(volatile uint64_t *)&mac_table[slot];

        /* Slot vuoto: fine della catena di probe, MAC non trovato */
        if (entry == 0)
            return -1;

        /* Trovato: valid bit acceso E MAC corrisponde */
        if ((entry & MAC_ENTRY_VALID_BIT) &&
            (entry & MAC_48BIT_MASK) == mac48) {
            return (int)((entry >> MAC_ENTRY_PORT_SHIFT) & 0x1);
        }

        slot = (slot + 1) & (MAC_TABLE_SIZE - 1);
    }
    return -1;  /* non trovato entro il limite di probe */
}

/* =========================================================================
 * KERNEL PRINCIPALE
 * =========================================================================
 *
 * Parametri kernel:
 *   rxq_gpu_p0 / rxq_gpu_p1: handle GPU per ricevere pacchetti (una per porta)
 *   txq_gpu_p0 / txq_gpu_p1: handle GPU per trasmettere pacchetti (una per porta)
 *   rxq0_mkey_for_txq1: chiave hardware che autorizza la NIC porta 1 a
 *       leggere il buffer GPU di porta 0 (per forwarding zero-copy p0→p1)
 *   rxq1_mkey_for_txq0: chiave hardware che autorizza la NIC porta 0 a
 *       leggere il buffer GPU di porta 1 (per forwarding zero-copy p1→p0)
 *   mac_table: hash table FIB in GPU global memory
 *   exit_cond: il CPU scrive 1 per fermare il kernel
 *   fwd_count: il kernel scrive il totale pacchetti forwardati (letto dal CPU)
 */
__global__ void bridge_kernel(
    struct doca_gpu_eth_rxq *rxq_gpu_p0,
    struct doca_gpu_eth_txq *txq_gpu_p0,
    struct doca_gpu_eth_rxq *rxq_gpu_p1,
    struct doca_gpu_eth_txq *txq_gpu_p1,
    uint32_t                 rxq0_mkey_for_txq1,
    uint32_t                 rxq1_mkey_for_txq0,
    uint64_t                *mac_table,
    uint32_t                *exit_cond,
    uint64_t                *fwd_count)
{
    /*
     * SHARED MEMORY — condivisa tra tutti i 32 thread del blocco.
     *
     * rx_first_pkt_idx: indice nel ring buffer del primo pacchetto del batch.
     *   Questo indice è "globale" nel ring: potrebbe valere es. 32768 dopo
     *   molti batch. doca_gpu_dev_eth_rxq_get_pkt_addr gestisce il wrap-around.
     *
     * rx_pkt_count: quanti pacchetti ci sono in questo batch (0..MAX_RX_NUM_PKTS).
     *
     * rx_attr[]: attributi di ogni pacchetto (almeno .bytes = lunghezza).
     *   Necessita shared memory perché doca_gpu_dev_eth_rxq_recv<BLOCK> la
     *   riempie in modo cooperativo: ogni thread scrive le proprie entry.
     *   Dopo __syncthreads(), tutti i thread possono leggere tutte le entry.
     *
     * fwd_decision[]: decisione di forward/drop per ogni pacchetto.
     *   Scritta nella fase parallela, letta da thread 0 nella fase seriale.
     *   Riusata tra le due direzioni (A e B) grazie a __syncthreads().
     */
    __shared__ uint64_t                          rx_first_pkt_idx;
    __shared__ uint32_t                          rx_pkt_count;
    __shared__ struct doca_gpu_dev_eth_rxq_attr  rx_attr[MAX_RX_NUM_PKTS];
    __shared__ uint8_t                           fwd_decision[MAX_RX_NUM_PKTS];

    /*
     * Contatori WQE e CQE per ciascuna TXQ.
     * Usati SOLO dal thread 0 (nelle fasi seriali di submit e poll).
     *
     * wqe_p1: prossimo indice WQE libero su txq1 (forwarding porta 0 → porta 1)
     * cqe_p1: prossimo indice CQE atteso su txq1
     * wqe_p0: prossimo indice WQE libero su txq0 (forwarding porta 1 → porta 0)
     * cqe_p0: prossimo indice CQE atteso su txq0
     *
     * I WQE sono in un ring: il firmware della NIC usa (wqe_idx % MAX_SQ_DESCR_NUM)
     * internamente. Noi teniamo il contatore monotonicamente crescente.
     */
    uint64_t wqe_p1 = 0, cqe_p1 = 0;
    uint64_t wqe_p0 = 0, cqe_p0 = 0;
    uint64_t tot_fwd = 0;

    doca_error_t ret;
    __syncthreads();  /* tutti i thread pronti prima di entrare nel loop */

    /*
     * LOOP PRINCIPALE: gira finché il CPU non scrive 1 in exit_cond.
     *
     * DOCA_GPUNETIO_VOLATILE(*exit_cond): legge exit_cond ogni iterazione
     * dalla global memory GPU, bypassando qualsiasi cache del compilatore.
     */
    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {

        /* ================================================================
         * DIREZIONE A: PORTA 0 → PORTA 1
         *
         * Riceviamo da rxq0 (pacchetti arrivati dalla rete su porta 0,
         * es. dall'Intel 810), processiamo con la FIB, inviamo su txq1
         * (verso la rete collegata alla porta 1 della BF2).
         * ================================================================ */

        /*
         * RICEZIONE (EXEC_SCOPE_BLOCK, tutti i 32 thread cooperano):
         *
         * Questa chiamata è bloccante: i 32 thread aspettano insieme
         * finché arrivano pacchetti OPPURE scade il timeout di 500 µs.
         * Se scade il timeout, rx_pkt_count = 0 e continuiamo il loop
         * (questo ci permette di controllare exit_cond regolarmente).
         *
         * Internamente DOCA divide il ring buffer tra i 32 thread:
         * thread t controlla le posizioni t, t+32, t+64, ...
         * del ring buffer per vedere se la NIC ha scritto nuovi pacchetti.
         * Quando tutti i thread hanno trovato i loro pacchetti (o è scaduto
         * il timeout), si sincronizzano e riempono rx_first_pkt_idx,
         * rx_pkt_count e rx_attr[].
         */
        ret = doca_gpu_dev_eth_rxq_recv<
                DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
                DOCA_GPUNETIO_ETH_MCST_AUTO,
                DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
                DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
            rxq_gpu_p0,
            MAX_RX_NUM_PKTS,
            MAX_RX_TIMEOUT_NS,
            &rx_first_pkt_idx,
            &rx_pkt_count,
            rx_attr);

        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0)
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            break;
        }

        if (rx_pkt_count > 0) {

            /* ============================================================
             * FASE PARALLELA — tutti i 32 thread elaborano in parallelo.
             *
             * Round-robin: thread t gestisce i pacchetti con indice
             *   t, t+32, t+64, t+96, ...
             * Se rx_pkt_count = 100:
             *   thread 0:  pacchetti 0, 32, 64, 96
             *   thread 1:  pacchetti 1, 33, 65, 97
             *   ...
             *   thread 3:  pacchetti 3, 35, 67, 99
             *   thread 4:  pacchetti 4, 36, 68  (99 < 100, 100 ≥ 100 → stop)
             *   ...
             * ============================================================ */
            uint32_t pkt_idx = threadIdx.x;
            while (pkt_idx < rx_pkt_count) {

                /*
                 * Ottieni l'indirizzo del pacchetto nel buffer GPU di porta 0.
                 * pkt_addr punta all'inizio dell'header Ethernet del pacchetto.
                 * rx_first_pkt_idx + pkt_idx = indice assoluto nel ring buffer.
                 * La funzione gestisce internamente il wrap-around del ring.
                 */
                uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                    rxq_gpu_p0, rx_first_pkt_idx + pkt_idx);
                const uint8_t *eth = (const uint8_t *)(uintptr_t)pkt_addr;

                /*
                 * Leggi dst_mac (byte 0-5) e src_mac (byte 6-11) dall'header.
                 *
                 * Layout header Ethernet IEEE 802.3:
                 *   [0..5]:   dst MAC  (destinatario del frame)
                 *   [6..11]:  src MAC  (mittente del frame)
                 *   [12..13]: ethertype (0x0800=IPv4, 0x0806=ARP, 0x86DD=IPv6, ...)
                 *   [14+]:    payload (IP, ARP, etc.)
                 *
                 * Il bridge non guarda ethertype o payload: lavora solo su MAC.
                 */
                uint64_t dst_mac48 = mac_to_u64(eth);       /* byte 0-5 */
                uint64_t src_mac48 = mac_to_u64(eth + 6);   /* byte 6-11 */

                /*
                 * BACKWARD LEARNING:
                 * "Questo frame è arrivato su porta 0 con src_mac X →
                 *  il dispositivo con MAC X si trova sulla porta 0."
                 *
                 * Aggiorniamo la FIB. La prossima volta che vediamo un
                 * pacchetto con dst_mac == X, lo inviamo sulla porta 0.
                 */
                mac_learn(mac_table, src_mac48, 0);

                /*
                 * FIB LOOKUP:
                 * "Dove devo inviare questo frame con dst_mac?"
                 *
                 *   -1: dst_mac sconosciuto → flood (manda all'altra porta)
                 *    0: dst_mac è su porta 0 → DROP (stessa porta di ingresso)
                 *    1: dst_mac è su porta 1 → FORWARD su porta 1
                 *
                 * Con 2 porte: "flood" = "manda a porta 1" = identico a forward.
                 * Quindi droppare solo se dst_port == 0 (= stessa porta di ingresso).
                 */
                int dst_port = mac_lookup(mac_table, dst_mac48);
                fwd_decision[pkt_idx] = (dst_port != 0) ? 1u : 0u;

                pkt_idx += blockDim.x;
            }
            /*
             * Barriera di sincronizzazione: aspetta che TUTTI i 32 thread
             * abbiano finito di scrivere in fwd_decision[].
             * Dopo questo punto, thread 0 può leggere tutte le decisioni.
             */
            __syncthreads();

            /* ============================================================
             * FASE SERIALE — solo thread 0 riempie i WQE.
             *
             * Perché serial e non parallelo?
             * I WQE sulla TXQ devono avere indici CONSECUTIVI senza buchi.
             * Non sappiamo in anticipo quali pacchetti vengono droppati,
             * quindi non possiamo pre-assegnare indici WQE in parallelo.
             * Thread 0 fa due passaggi:
             *   1. Conta quanti pacchetti vanno forwardati (n_to_forward)
             *   2. Assegna WQE consecutivi: wqe_p1+0, wqe_p1+1, ..., wqe_p1+(n-1)
             * ============================================================ */
            if (threadIdx.x == 0) {

                /* Passaggio 1: conta i forward */
                uint32_t n_to_forward = 0;
                for (uint32_t i = 0; i < rx_pkt_count; i++) {
                    if (fwd_decision[i])
                        n_to_forward++;
                }

                if (n_to_forward > 0) {

                    /* Passaggio 2: riempi WQE per ogni pacchetto forwardato */
                    uint32_t wqe_offset = 0;
                    for (uint32_t i = 0; i < rx_pkt_count; i++) {
                        if (!fwd_decision[i])
                            continue;

                        /* Indirizzo del pacchetto in GPU memory di porta 0 */
                        uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                            rxq_gpu_p0, rx_first_pkt_idx + i);

                        /*
                         * Flag NOTIFY: solo sull'ULTIMO WQE del batch.
                         *
                         * La NIC genera una CQE (Completion Queue Entry)
                         * SOLO per i WQE con flag NOTIFY. Così invece di
                         * ricevere una CQE per ogni dei 2048 pacchetti
                         * (enorme overhead), ne riceviamo UNA per batch.
                         * Questo è il pattern consigliato da NVIDIA.
                         */
                        enum doca_gpu_eth_send_flags flags =
                            (wqe_offset == n_to_forward - 1) ?
                                DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY :
                                DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;

                        /*
                         * ZERO-COPY CROSS-PORT WQE:
                         *
                         * Il WQE dice alla NIC porta 1:
                         *   "Leggi rx_attr[i].bytes byte dall'indirizzo pkt_addr
                         *    (che si trova nel buffer GPU della porta 0)
                         *    usando il mkey rxq0_mkey_for_txq1,
                         *    e trasmetti quei byte in rete."
                         *
                         * La NIC porta 1 accede alla memoria GPU della porta 0
                         * via PCIe senza passare per la CPU. Zero copie.
                         *
                         * rxq0_mkey_for_txq1: il mkey ottenuto da
                         *   doca_mmap_get_mkey(mmap_porta0, ddev_porta1)
                         * È la chiave RDMA che autorizza la NIC porta 1 ad
                         * accedere alla memoria GPU registrata per porta 0.
                         */
                        struct doca_gpu_dev_eth_txq_wqe *wqe =
                            doca_gpu_dev_eth_txq_get_wqe_ptr(
                                txq_gpu_p1, wqe_p1 + wqe_offset);

                        doca_gpu_dev_eth_txq_wqe_prepare_send(
                            txq_gpu_p1,
                            wqe,
                            wqe_p1 + wqe_offset,
                            pkt_addr,
                            rxq0_mkey_for_txq1,
                            rx_attr[i].bytes,
                            flags);

                        wqe_offset++;
                    }

                    /*
                     * SUBMIT (doorbell):
                     * Scrive nell'UAR (User Access Region) della BF2 tramite PCIe.
                     * Questo dice alla NIC porta 1 "ci sono nuovi WQE da processare
                     * fino all'indice wqe_p1 + n_to_forward".
                     * La NIC inizia immediatamente il DMA dalla GPU e la trasmissione.
                     */
                    doca_gpu_dev_eth_txq_submit(txq_gpu_p1, wqe_p1 + n_to_forward);

                    /*
                     * POLL COMPLETION (busy-wait):
                     * Thread 0 legge in loop la CQE in GPU memory finché la NIC
                     * la scrive (dopo aver processato il WQE con NOTIFY).
                     *
                     * RESOURCE_SHARING_MODE_GPU: la TXQ è condivisa solo in GPU.
                     * SYNC_SCOPE_CTA: sincronizzazione a livello di blocco CUDA.
                     * WAIT_FLAG_B: attesa bloccante (busy-wait finché CQE arriva).
                     */
                    ret = doca_gpu_dev_eth_txq_poll_completion_at<
                            DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
                        txq_gpu_p1,
                        cqe_p1,
                        DOCA_GPUNETIO_ETH_WAIT_FLAG_B);

                    if (ret != DOCA_SUCCESS)
                        DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;

                    wqe_p1  += n_to_forward;
                    cqe_p1  += 1;
                    tot_fwd += n_to_forward;
                }
            }
            /* Aspetta thread 0 prima di riusare rx_attr e fwd_decision */
            __syncthreads();
        }

        /* Controlla exit_cond prima di passare alla direzione B */
        if (DOCA_GPUNETIO_VOLATILE(*exit_cond) != 0)
            break;

        /* ================================================================
         * DIREZIONE B: PORTA 1 → PORTA 0
         *
         * Speculare alla direzione A, con porte invertite:
         *   - ricezione da rxq1
         *   - backward learning impara "src_mac è su porta 1"
         *   - drop se dst_port == 1 (stessa porta di ingresso)
         *   - forward su txq0 con mkey rxq1_mkey_for_txq0
         * ================================================================ */

        ret = doca_gpu_dev_eth_rxq_recv<
                DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
                DOCA_GPUNETIO_ETH_MCST_AUTO,
                DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
                DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
            rxq_gpu_p1,
            MAX_RX_NUM_PKTS,
            MAX_RX_TIMEOUT_NS,
            &rx_first_pkt_idx,
            &rx_pkt_count,
            rx_attr);

        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0)
                DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            break;
        }

        if (rx_pkt_count > 0) {

            uint32_t pkt_idx = threadIdx.x;
            while (pkt_idx < rx_pkt_count) {

                uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                    rxq_gpu_p1, rx_first_pkt_idx + pkt_idx);
                const uint8_t *eth = (const uint8_t *)(uintptr_t)pkt_addr;

                uint64_t dst_mac48 = mac_to_u64(eth);
                uint64_t src_mac48 = mac_to_u64(eth + 6);

                /* Backward learning: il mittente è raggiungibile su porta 1 */
                mac_learn(mac_table, src_mac48, 1);

                /* FIB lookup: dove va il destinatario? */
                int dst_port = mac_lookup(mac_table, dst_mac48);

                /* Drop se dst è sulla stessa porta 1, forward altrimenti */
                fwd_decision[pkt_idx] = (dst_port != 1) ? 1u : 0u;

                pkt_idx += blockDim.x;
            }
            __syncthreads();

            if (threadIdx.x == 0) {

                uint32_t n_to_forward = 0;
                for (uint32_t i = 0; i < rx_pkt_count; i++) {
                    if (fwd_decision[i])
                        n_to_forward++;
                }

                if (n_to_forward > 0) {

                    uint32_t wqe_offset = 0;
                    for (uint32_t i = 0; i < rx_pkt_count; i++) {
                        if (!fwd_decision[i])
                            continue;

                        uint64_t pkt_addr = doca_gpu_dev_eth_rxq_get_pkt_addr(
                            rxq_gpu_p1, rx_first_pkt_idx + i);

                        enum doca_gpu_eth_send_flags flags =
                            (wqe_offset == n_to_forward - 1) ?
                                DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY :
                                DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;

                        /* WQE zero-copy: txq0 legge dal buffer GPU di rxq1 */
                        struct doca_gpu_dev_eth_txq_wqe *wqe =
                            doca_gpu_dev_eth_txq_get_wqe_ptr(
                                txq_gpu_p0, wqe_p0 + wqe_offset);

                        doca_gpu_dev_eth_txq_wqe_prepare_send(
                            txq_gpu_p0,
                            wqe,
                            wqe_p0 + wqe_offset,
                            pkt_addr,
                            rxq1_mkey_for_txq0,
                            rx_attr[i].bytes,
                            flags);

                        wqe_offset++;
                    }

                    doca_gpu_dev_eth_txq_submit(txq_gpu_p0, wqe_p0 + n_to_forward);

                    ret = doca_gpu_dev_eth_txq_poll_completion_at<
                            DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
                            DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
                        txq_gpu_p0,
                        cqe_p0,
                        DOCA_GPUNETIO_ETH_WAIT_FLAG_B);

                    if (ret != DOCA_SUCCESS)
                        DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;

                    wqe_p0  += n_to_forward;
                    cqe_p0  += 1;
                    tot_fwd += n_to_forward;
                }
            }
            __syncthreads();
        }

    } /* fine while loop principale */

    /*
     * Scrivi il contatore finale in CPU_GPU memory.
     * __threadfence_system(): flush le cache L1/L2 della GPU attraverso
     * il bus PCIe in modo che la CPU veda il valore aggiornato dopo
     * cudaStreamSynchronize(). Senza questo, la CPU potrebbe leggere
     * un valore stale dalla sua cache.
     */
    if (threadIdx.x == 0) {
        *fwd_count = tot_fwd;
        __threadfence_system();
    }
}

/* =========================================================================
 * WRAPPER EXTERN "C" — chiamabile da codice C (gpu_bridge.c)
 * =========================================================================
 * Il kernel CUDA ha C++ name mangling. extern "C" produce un simbolo
 * con nome C semplice, linkabile dal codice C compilato con gcc.
 */
extern "C" {

doca_error_t kernel_launch_bridge(cudaStream_t stream,
                                   struct bridge_kernel_params *kp)
{
    cudaError_t cuda_ret;

    if (!kp || !kp->mac_table || !kp->exit_cond || !kp->fwd_count)
        return DOCA_ERROR_INVALID_VALUE;

    /* Verifica che non ci siano errori CUDA pendenti dal setup precedente */
    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA error before kernel launch: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    /*
     * Lancio del kernel: <<<1 blocco, 32 thread, 0 shared_mem dinamica, stream>>>
     *
     * 1 blocco: il kernel è persistente (gira per sempre su un solo blocco).
     * 32 thread: esattamente 1 warp, richiesto da EXEC_SCOPE_BLOCK.
     * 0: la shared memory è dichiarata staticamente con __shared__ nel kernel
     *    (dimensione fissa a compile time, non serve specifica dinamica).
     * stream: CUDA stream non-bloccante, il CPU continua mentre la GPU lavora.
     */
    bridge_kernel<<<1, BRIDGE_BLOCK_THREADS, 0, stream>>>(
        kp->rxq_gpu[0],
        kp->txq_gpu[0],
        kp->rxq_gpu[1],
        kp->txq_gpu[1],
        kp->rxq_mkey_for_other[0],
        kp->rxq_mkey_for_other[1],
        kp->mac_table,
        kp->exit_cond,
        kp->fwd_count);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n",
                cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern "C" */
