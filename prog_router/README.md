# GPU L3 Router

Un router IPv4 implementato con NVIDIA DOCA GPUNetIO: i pacchetti vengono
DMA-ati direttamente nella GPU dalla BlueField-2, instradati dal kernel
CUDA in base a una tabella di routing statica, e ritrasmessi in rete —
senza che la CPU tocchi mai un byte di payload.

È il "fratello L3" di [`prog_gpu_bridge`](../prog_gpu_bridge/README.md)
(bridge L2): stesso hardware, stesso datapath GPUNetIO di base, ma qui la
decisione di forwarding si basa sull'**IP di destinazione** invece che sul
MAC, e il pacchetto viene effettivamente **modificato** ad ogni hop (come
impone lo standard IP), non solo inoltrato.

---

## Cos'è un router L3 (e come differisce da un bridge L2)?

Un **bridge** (vedi `prog_gpu_bridge`) collega due segmenti della STESSA
rete: guarda i MAC address e non tocca mai il pacchetto.

Un **router** collega reti DIVERSE: guarda l'**indirizzo IP di
destinazione**, decide su quale interfaccia instradarlo confrontandolo con
una tabella di rotte, e — a differenza di un bridge — è tenuto dallo
standard IP a **modificare** il pacchetto ad ogni hop:

1. decrementa il **TTL** (Time To Live) di 1 (evita che un pacchetto
   in loop giri per sempre nella rete);
2. ricalcola il **checksum** dell'header IPv4 (cambia perché è cambiato
   il TTL, uno dei campi che il checksum protegge);
3. riscrive **l'header Ethernet**: il MAC sorgente diventa quello della
   PROPRIA interfaccia di uscita (non più quello del mittente originale),
   il MAC destinazione diventa quello del **prossimo salto** (next-hop) —
   che sia un altro router o l'host finale.

```
[Rete 10.0.0.0/24] ──── [porta 0] ~~~ ROUTER ~~~ [porta 1] ──── [Rete 10.0.1.0/24]
                                         │
                                    [porta N] ──── [altra rete...]
```

---

## RIB e FIB: due nomi, due scopi

L'utente di questo progetto conosceva già i due acronimi ma non era
sicuro di quando si applicano — ecco la distinzione usata in questo
programma (coerente con la terminologia standard del networking):

- **RIB** (Routing Information Base): l'elenco "grezzo" delle rotte, così
  come configurate o apprese da un protocollo di routing (BGP, OSPF, ...).
  In questo programma: le righe del file `-r routes.txt`, lette una per
  una da `load_routes_file()` in `gpu_router.c`.
- **FIB** (Forwarding Information Base): la tabella *compilata*, ottimizzata
  per la ricerca ad alta velocità nel data plane. Qui: lo stesso array
  della RIB, ma **ordinato per lunghezza di prefisso decrescente** e
  copiato in memoria GPU — è quello che il kernel CUDA consulta per ogni
  pacchetto.

Un router "vero" (con BGP/OSPF) avrebbe una RIB enorme e dinamica, ricompilata
in una FIB periodicamente. Questo programma non implementa alcun protocollo
di routing dinamico: la RIB è statica (un file di testo, caricato una sola
volta all'avvio) — per questo RIB e FIB qui hanno lo stesso *contenuto*,
cambia solo la *struttura dati* (array non ordinato in RAM host vs. array
ordinato in VRAM GPU).

---

## Longest Prefix Match (LPM)

Per ogni pacchetto, il router cerca nella FIB la rotta più **specifica**
che copre l'IP di destinazione. Esempio: con le rotte

```
10.0.0.0/24   -> porta 0
0.0.0.0/0     -> porta 1   (default route)
```

un pacchetto per `10.0.0.42` fa match su ENTRAMBE le rotte (`10.0.0.0/24`
E `0.0.0.0/0`), ma va scelta quella con il prefisso più lungo (`/24`,
più specifica) — cioè porta 0, non la default route.

**Come lo implementiamo:** la FIB viene ordinata dalla CPU (una volta sola,
all'avvio) per lunghezza di prefisso decrescente. Il kernel CUDA scandisce
l'array linearmente e si ferma al PRIMO match: per costruzione, essendo
l'array ordinato dal prefisso più lungo al più corto, il primo match trovato
è sempre il Longest Prefix Match. Vedi `fib_lookup()` in
`gpu_router_kernel.cu` per i dettagli e il perché di questa scelta (semplice
scansione lineare) invece di una trie/Patricia trie (usata dai router
hardware con tabelle BGP da centinaia di migliaia di rotte — qui, con al
più `MAX_ROUTES=1024` rotte, la scansione lineare è già ampiamente
sufficiente e molto più semplice da leggere/verificare).

---

## Nessun ARP, nessun ICMP: cosa NON fa questo router

Per restare focalizzati sul data plane GPU (l'obiettivo didattico del
progetto), questo router **non implementa**:

- **ARP**: non risponde a richieste ARP per i propri IP, non fa la
  risoluzione IP→MAC per i next-hop. Il MAC di ogni next-hop è **statico**,
  fornito nel file di rotte (l'equivalente di una voce ARP permanente).
  **Implicazione pratica**: gli host/router collegati a ciascuna porta
  devono avere una voce ARP statica per l'IP del router, puntata al MAC
  reale di quella porta (stampato all'avvio del programma, es.
  `NIC porta 0 aperta: ad:00.0  MAC ...`). Su Linux:
  `sudo ip neigh replace <ip_router> lladdr <mac_porta> dev <if> nud permanent`.
- **ICMP** (Time Exceeded quando la TTL scade, Redirect, Destination
  Unreachable quando manca una rotta): i pacchetti vengono scartati
  silenziosamente, con un contatore diagnostico dedicato (vedi sotto), ma
  nessun messaggio ICMP viene generato verso il mittente.
- **Local delivery**: pacchetti destinati a un IP del router stesso (se
  gli si assegnasse un IP) non vengono gestiti in modo speciale — semplicemente
  non ci sarà una rotta che li copre (a meno che l'admin non ne configuri una),
  e verranno scartati come "no route".
- **Frammentazione IP, opzioni IP con azioni particolari, IPv6**: fuori
  scope. Il campo IHL viene comunque rispettato (l'header può avere
  opzioni: il checksum viene ricalcolato sull'header intero, `ihl*4` byte).

Questi sono tutti **punti di estensione naturali**, non limiti della GPU:
il kernel CUDA potrebbe generare pacchetti ICMP/ARP con lo stesso
meccanismo di TX già usato per il forwarding. Sono stati lasciati fuori
per mantenere il kernel comprensibile e la sua logica di decisione (il
cuore didattico del progetto) ben visibile, invece di annegarla in una
gestione di casi speciali.

---

## Architettura GPU Zero-Copy

Identica nell'impalcatura al bridge (stesso hardware BF2 + A30X, stessa
API DOCA GPUNetIO):

```
                    ┌──────────────────────────────────┐
                    │            GPU A30X               │
                    │                                   │
[Rete porta 0] ─DMA→│ rxq0_buf ─kernel CUDA─→ riscrive  │─DMA→ [Rete porta 1]
                    │ (FIB lookup, TTL--, checksum,     │
                    │  MAC src/dst) ────────→ WQE       │
[Rete porta 1] ─DMA→│ rxq1_buf ─kernel CUDA─→ riscrive  │─DMA→ [Rete porta 0]
                    │                                   │
                    └──────────────────────────────────┘
```

**Zero-copy cross-port:** come nel bridge, il buffer GPU della porta
sorgente è registrato nel Protection Domain RDMA di TUTTE le altre porte
(vedi `setup_port_rxq` in `gpu_router.c`): la NIC di uscita fa DMA READ
direttamente da lì, senza copie intermedie. La differenza rispetto al
bridge è che qui il kernel **modifica** quel buffer (TTL, checksum, MAC)
PRIMA che il WQE venga sottomesso alla NIC — la riscrittura è quindi
"gratis" dal punto di vista della banda: nessun buffer aggiuntivo, nessuna
copia, solo qualche decina di byte scritti in VRAM che erano già lì.

**DOCA Flow — filtro EtherType IPv4 in hardware:** a differenza del
bridge (che accetta qualunque frame L2 con un match wildcard), qui la
pipe DOCA Flow di ogni porta accetta **solo** EtherType `0x0800` (IPv4):
tutto il resto (ARP, IPv6, ecc.) viene scartato direttamente dalla NIC,
prima ancora di arrivare alla GPU. Vedi `setup_port_flow()`.

---

## Il Kernel CUDA — cosa succede per ogni pacchetto

```
while (!exit_cond):
    per ogni porta sorgente src:
        ricevi fino a MAX_RX_NUM_PKTS pacchetti (timeout 500 µs)
        [32 thread in parallelo] per ogni pacchetto ricevuto:
            valida IPv4 (version==4, ihl>=5)         -> altrimenti DROP (malformed)
            ttl <= 1?                                 -> DROP (ttl_expired)
            FIB lookup (Longest Prefix Match su dst)  -> altrimenti DROP (no_route)
            altrimenti:
                ttl -= 1
                ricalcola checksum IPv4
                mac_sorgente  = MAC della porta di uscita
                mac_destinaz. = next-hop della rotta
                accoda un WQE sulla TXQ della porta di uscita
        [solo thread 0] per ogni TXQ con WQE in questo batch:
            fixup NOTIFY sull'ultimo WQE, submit (doorbell), poll CQE
```

A differenza del bridge, che può generare fino a N-1 WQE per un singolo
pacchetto floodato, qui **ogni pacchetto genera al più un solo WQE**: il
routing IP unicast non floода mai (vedi il commento esteso in testa a
`gpu_router_kernel.cu` sul perché la "egress mask" del bridge si riduce
a un singolo intero `eg_port` qui).

---

## Formato del file di rotte

Vedi [`routes.example.txt`](routes.example.txt) per un esempio completo e
commentato. In sintesi, una rotta per riga:

```
<rete>/<prefisso>   <porta_uscita>   <mac_next_hop>
```

- `<rete>/<prefisso>`: rete di destinazione in CIDR (es. `10.0.0.0/24`,
  `0.0.0.0/0` per la default route).
- `<porta_uscita>`: indice 0-based, nello stesso ordine dei flag `-n`
  sulla command line (il primo `-n` è la porta 0, ecc.).
- `<mac_next_hop>`: MAC a cui riscrivere il pacchetto in uscita (equivalente
  a una voce ARP statica — vedi sopra "Nessun ARP").

Righe vuote e che iniziano con `#` sono ignorate. Le rotte vengono
validate riga per riga (IP valido, prefisso in `[0,32]`, porta esistente,
MAC valido) con messaggi di errore che indicano il numero di riga.

---

## Setup dell'ambiente

Il progetto richiede le dipendenze descritte in `../DEPENDENCIES.md`
(le stesse del bridge: DOCA 3.3, CUDA, driver BlueField-2 in modalità NIC).

Il namespace di rete `bf2` è necessario perché DOCA usa `libibverbs` per
scoprire i device RDMA della BF2, visibili solo nel namespace dove vivono
le sue interfacce di rete.

---

## Compilazione

```bash
cd /home/prognose/doca-gpunetio/prog_router
make
```

---

## Esecuzione

```bash
sudo ip netns exec bf2 ./gpu_router \
    -n ad:00.0 \
    -n ad:00.1 \
    -g b0:00.0 \
    -r routes.txt
```

- `-n ad:00.0`: prima porta BF2 (porta 0 del router)
- `-n ad:00.1`: seconda porta BF2 (porta 1 del router) — ripetibile per
  router con più di 2 porte
- `-g b0:00.0`: GPU A30X
- `-r routes.txt`: file di rotte statiche (vedi `routes.example.txt`)
- `-i <n>`: indice CUDA device, default 0

### Test

1. Copia e adatta `routes.example.txt` con gli IP/MAC reali della tua
   topologia di test.
2. Configura sugli host di test una voce ARP statica per l'IP del router,
   puntata al MAC di interfaccia stampato all'avvio (vedi sopra "Nessun ARP").
3. Configura sugli host di test una rotta verso l'altra sottorete che
   passi per l'IP di questo router.
4. Invia traffico IP tra le due sottoreti e verifica con `tcpdump`
   sull'interfaccia di uscita che i pacchetti arrivino con TTL decrementata
   di 1 e i MAC L2 riscritti correttamente.

---

## Output atteso

```
NIC porta 0 aperta: ad:00.0  MAC ...
NIC porta 1 aperta: ad:00.1  MAC ...
Caricate 3 rotte da 'routes.txt' (ordinate per prefisso decrescente):
  10.0.0.0           /24 -> porta 0  next-hop aa:bb:cc:dd:ee:01
  10.0.1.0           /24 -> porta 1  next-hop aa:bb:cc:dd:ee:02
  0.0.0.0            / 0 -> porta 1  next-hop aa:bb:cc:dd:ee:ff
[porta 0] RXQ pronta — buf 0x... size 33554432 B  MAC ...
[porta 1] RXQ pronta — ...
[porta 0] TXQ pronta — 1024 WQE
[porta 1] TXQ pronta — 1024 WQE
[porta 0] DOCA Flow pronto — BASIC ROOT match EtherType IPv4 → RSS queue 0 (resto: DROP)
[porta 1] DOCA Flow pronto — BASIC ROOT match EtherType IPv4 → RSS queue 0 (resto: DROP)

GPU Router a 2 porte avviato:
  porta 0: ad:00.0  MAC ...
  porta 1: ad:00.1  MAC ...
  GPU:     b0:00.0 (CUDA device 0)
  FIB:     3 rotte (LPM per scansione lineare ordinata)
Premi Ctrl+C per fermare.

[live] rx_tot=120 (porta0=80 porta1=40)  fwd=118  drop_no_route=1  drop_ttl_expired=0  drop_malformed=1

^C
Fermato. Totale pacchetti instradati: 118
  Ricevuti per porta: porta0=80 porta1=40
  drop_no_route=1  drop_ttl_expired=0  drop_malformed=1
Done.
```

---

## Riferimenti

- Struct pacchetto (`ether_hdr`, `ipv4_hdr`) e convenzioni di parsing sul
  buffer GPU: conformi al sample ufficiale NVIDIA DOCA
  `gpu_packet_processing` (`/opt/mellanox/doca/applications/gpu_packet_processing`).
- Match DOCA Flow su EtherType IPv4 (`outer.eth.type` = `DOCA_FLOW_ETHER_TYPE_IPV4`):
  stessa costante usata in `gpu_packet_processing/config_queues/flow.c`. Quel
  sample imposta anche `outer.l3_type` insieme a `outer.eth.type`, ma lo fa su
  una pipe `DOCA_FLOW_PIPE_CONTROL`; su una pipe `DOCA_FLOW_PIPE_BASIC` come
  quella di questo router, impostare entrambi i campi insieme fa fallire
  `doca_flow_pipe_create` con `DOCA_ERROR_INVALID_VALUE` (verificato su questo
  hardware/firmware). Il solo match su `eth.type` è stato quindi verificato
  funzionante end-to-end ed è comunque sufficiente: l'EtherType 0x0800
  identifica IPv4 senza ambiguità.
- Datapath RXQ/TXQ GPU (mmap cross-port, WQE, NOTIFY/CQE): stesso schema
  verificato funzionante in [`../prog_gpu_bridge`](../prog_gpu_bridge/README.md)
  su questo stesso hardware (BlueField-2 in modalità NIC + A30X).
- L'intero avvio (apertura NIC/GPU, RXQ/TXQ, DOCA Flow, kernel CUDA) è stato
  testato su questa macchina (BlueField-2 in modalità NIC, `ad:00.0`/`ad:00.1`,
  GPU A30X `b0:00.0`): il programma si inizializza e gira correttamente in
  attesa di pacchetti. Il forwarding effettivo di traffico reale richiede
  comunque host configurati con le voci ARP statiche descritte sopra.
