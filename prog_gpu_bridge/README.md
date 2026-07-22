# GPU L2 Bridge

Un bridge L2 implementato con NVIDIA DOCA GPUNetIO: i pacchetti vengono
DMA-ati direttamente nella GPU dalla BF2, processati dal kernel CUDA, e
ritrasferiti in rete — senza che la CPU tocchi nemmeno un byte di payload.

---

## Cos'è un bridge L2?

Un **bridge L2** (livello 2 del modello OSI) connette due segmenti di rete
a livello Ethernet. A differenza di un router (che lavora su IP), il bridge
lavora sui **MAC address** (il "nome" fisico di ogni scheda di rete).

Immagina di avere due gruppi di computer:

```
[Intel 810] ──────── [BF2 porta 0] ~~~ BRIDGE ~~~ [BF2 porta 1] ──────── [altro host]
   10.0.0.1                                                                   10.0.0.2
```

Il bridge decide autonomamente, per ogni frame Ethernet che riceve, su quale
porta "uscire". Per farlo, usa due meccanismi.

---

## Backward Learning (apprendimento inverso)

Il bridge parte da zero: non sa dove si trovano i dispositivi.

Ogni volta che arriva un frame, il bridge legge il **src_mac** (MAC del mittente)
e lo registra nella **FIB** (Forward Information Base, una tabella hash):

```
"Il frame è arrivato su porta 0 con src_mac AA:BB:CC:DD:EE:FF →
 d'ora in poi, chi vuole raggiungere AA:BB:CC:DD:EE:FF deve usare porta 0."
```

Questo si chiama "backward" (inverso) perché si impara dal mittente, non
dal destinatario. Progressivamente la FIB si riempie, e il bridge diventa
sempre più preciso.

---

## FIB Table Lookup e Decisione di Forward/Drop

Per ogni frame ricevuto, il bridge cerca il **dst_mac** (MAC del destinatario)
nella FIB:

| Risultato lookup | Azione |
|------------------|--------|
| MAC trovato, porta = porta ingresso | **DROP**: il destinatario è già sullo stesso segmento, non serve forwardare |
| MAC trovato, porta = altra porta | **FORWARD** sull'altra porta |
| MAC non trovato | **FLOOD**: invia sull'altra porta (con 2 porte, flood = forward) |

Il caso DROP è fondamentale: evita di rimandare un frame sul segmento da
cui è già arrivato, creando traffico inutile.

---

## Architettura GPU Zero-Copy

```
                    ┌─────────────────────────────────┐
                    │          GPU A30X                │
                    │                                  │
[Rete porta 0] ─DMA→│ rxq0_buf ──kernel CUDA──→ WQE  │─DMA→ [Rete porta 1]
                    │ (MAC table: learn + lookup)      │
                    │                                  │
[Rete porta 1] ─DMA→│ rxq1_buf ──kernel CUDA──→ WQE  │─DMA→ [Rete porta 0]
                    │                                  │
                    └─────────────────────────────────┘
```

**Perché la GPU?**

La GPU ha migliaia di core che lavorano in parallelo. Con DOCA GPUNetIO,
la NIC BlueField-2 fa DMA direttamente nella memoria GPU (bypassa la CPU
e la RAM di sistema). Il kernel CUDA elabora i pacchetti in memoria GPU
e scrive istruzioni di trasmissione (WQE) sempre in GPU memory. La NIC
legge i WQE e trasmette — di nuovo DMA diretto dalla GPU. La CPU non
tocca mai i dati.

**Zero-copy cross-port:**

Quando si forwarda dalla porta 0 alla porta 1, i dati in `rxq0_buf`
(memoria GPU) vengono letti **direttamente** dalla NIC porta 1 tramite il
suo WQE — senza copiare in un buffer intermedio. Questo richiede che il
buffer GPU di porta 0 sia registrato anche nel Protection Domain RDMA di
porta 1 (vedi sezione "Mkey cross-port").

---

## Il Kernel CUDA

Un singolo kernel persistente, **1 blocco, 32 thread** (= 1 warp NVIDIA).

### Perché 32 thread?

`EXEC_SCOPE_BLOCK` richiede che tutti i thread del blocco cooperino per la
ricezione. Con 32 thread = 1 warp, la sincronizzazione è garantita a livello
hardware (tutti i thread del warp eseguono la stessa istruzione in parallelo)
senza overhead di `__syncthreads` sul percorso di ricezione.

### Loop principale

```
while (!exit_cond):
    ├── Ricevi da rxq0 (fino a 2048 pacchetti, timeout 500 µs)
    │   ├── [32 thread in parallelo] per ogni pacchetto:
    │   │       leggi src_mac, dst_mac dall'header Ethernet
    │   │       mac_learn(src_mac, porta=0)
    │   │       dst_port = mac_lookup(dst_mac)
    │   │       fwd_decision[i] = (dst_port != 0) ? FORWARD : DROP
    │   └── [solo thread 0] riempi WQE su txq1, submit, aspetta CQE
    │
    └── Ricevi da rxq1 (speculare, porta 1 → porta 0)
```

### Fase parallela vs seriale

**Fase parallela** (tutti i 32 thread): ogni thread processa i propri
pacchetti in round-robin (thread 0: pacchetti 0, 32, 64...; thread 1:
pacchetti 1, 33, 65...). Hash table atomica: `atomicCAS` per inserire,
`atomicExch` per aggiornare.

**Fase seriale** (solo thread 0): riempi i WQE. Deve essere seriale perché
i WQE devono avere indici **consecutivi senza buchi** — non possiamo
pre-assegnare indici in parallelo senza sapere quanti pacchetti verranno
droppati.

---

## MAC Table (FIB)

Hash table in memoria GPU: `uint64_t mac_table[4096]`.

**Formato di ogni entry:**
```
bit 63:   valid (1 = slot occupato)
bit 48:   porta (0 o 1)
bit 0-47: MAC address a 48 bit
```

**Hash:** FNV-1a a 6 iterazioni (una per byte MAC). Fast, no divisione,
buona distribuzione. Risultato mascherato con `& 4095` (potenza di 2).

**Collision resolution:** linear probing fino a 256 slot. Con 4096 slot
e un numero realistico di MAC (< 100), la catena di probe è raramente
più lunga di 2-3.

**Concorrenza:** `atomicCAS` per inserire in slot vuoto, `atomicExch`
per aggiornare la porta. Se due thread cercano di inserire lo stesso MAC,
uno vince il CAS e l'altro trova il MAC già presente e aggiorna. Nessun
deadlock, nessuna corruzione.

---

## Mkey Cross-Port (dettaglio tecnico)

In InfiniBand/RDMA (il layer usato internamente da DOCA), ogni NIC ha un
**Protection Domain (PD)** separato. Una registrazione di memoria (come
il buffer GPU della RXQ) produce un **LKEY** valido solo per quel PD.

Per il forwarding cross-port zero-copy, il buffer GPU di porta 0 deve
essere accessibile dalla NIC porta 1. Soluzione:

```c
doca_mmap_add_dev(mmap0, ddev0);   // porta 0 può fare DMA write (RX)
doca_mmap_add_dev(mmap0, ddev1);   // porta 1 può fare DMA read (TX)
doca_mmap_start(mmap0);            // registra con entrambi i PD

doca_mmap_get_mkey(mmap0, ddev1, &mkey_for_txq1);  // LKEY per porta 1
```

Il kernel CUDA usa `mkey_for_txq1` nei WQE della TXQ1 — è la "firma"
che autorizza la NIC porta 1 a leggere quella memoria GPU.

---

## Setup dell'ambiente

Il progetto richiede le dipendenze descritte in `../DEPENDENCIES.md`.

Il namespace di rete `bf2` è necessario perché DOCA usa `libibverbs` per
scoprire i device RDMA della BF2, che sono visibili solo nel namespace
dove vivono le interfacce di rete.

---

## Compilazione

```bash
cd /home/prognose/doca-gpunetio/gpu_bridge
make
```

---

## Esecuzione

```bash
sudo ip netns exec bf2 ./gpu_bridge \
    -n ad:00.0 \
    -n ad:00.1 \
    -g b0:00.0
```

- `-n ad:00.0`: prima porta BF2 (porta 0 del bridge)
- `-n ad:00.1`: seconda porta BF2 (porta 1 del bridge)
- `-g b0:00.0`: GPU A30X

### Test con traffico dall'Intel 810

Invia traffico dall'Intel 810 verso la porta 0 della BF2. Il bridge
imparerà il MAC dell'Intel 810 sulla porta 0. Se c'è un host collegato
alla porta 1, il traffico viene forwardato lì (e viceversa).

Per verificare che i pacchetti vengano forwardati, usa `tcpdump` su
un'interfaccia collegata alla porta 1 della BF2.

---

## Output atteso

```
NIC ad:00.0 aperta (porta 0)
NIC ad:00.1 aperta (porta 1)
[porta 0] RXQ pronta — buf 0x... size 33554432  mkey_self 0x...  mkey_other 0x...
[porta 1] RXQ pronta — ...
[porta 0] TXQ pronta — 1024 WQE
[porta 1] TXQ pronta — 1024 WQE
[porta 0] DOCA Flow pronto — BASIC ROOT pipe wildcard → RSS queue 0
[porta 1] DOCA Flow pronto — BASIC ROOT pipe wildcard → RSS queue 0

GPU Bridge avviato:
  Porta 0: ad:00.0
  Porta 1: ad:00.1
  GPU:     b0:00.0 (CUDA device 0)
  MAC table: 4096 slot (FNV-1a hash, linear probing)
Premi Ctrl+C per fermare.

^C
Fermato. Totale pacchetti forwardati: 42
Done.
```
