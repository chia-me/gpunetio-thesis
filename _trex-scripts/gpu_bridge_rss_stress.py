"""
Profilo stateless T-Rex per il GPU L2 Bridge RSS (prog_gpu_bridge_rss).

Obiettivo: generare traffico con ENTROPIA sui campi che la NIC BF2 usa per
l'hash RSS (IPv4 src/dst + UDP src port — vedi RSS_HASH_FIELDS in
gpu_bridge_rss.h: DOCA_FLOW_RSS_IPV4|IPV6|UDP|TCP), così da distribuire i
pacchetti su tutte le N_QUEUES_PER_PORT code RX invece che su una sola.

Differenza rispetto a gpu_bridge_stress.py / gpu_forwarder_stress.py: quegli
script usano un'unica 5-tupla FISSA (giusto per v1/il forwarder, che hanno
1 sola coda). Qui invece src IP, dst IP e sport UDP variano pacchetto per
pacchetto tramite il Field Engine di T-Rex (STLScVmRaw/STLVmFlowVar), così
l'hash Toeplitz della NIC vede input diversi e sceglie code RX diverse.

Topologia (porte fisiche, vedi /etc/trex_cfg.yaml):
    interfaces: ["2a:00.0", "2c:00.0"] -> porta logica 0 = 2a:00.0 (dummy,
    non collegata), porta logica 1 = 2c:00.0 (reale, verso la BF2).
    Il TX va SEMPRE fatto da -p 1 (stessa convenzione di gpu_bridge_stress.py).
    dst_mac NON specificato esplicitamente, stessa nota/motivazione di
    gpu_bridge_stress.py: lasciarlo risolvere a T-Rex.

Verifica lato bridge: mentre questo profilo gira, guarda su gpu_bridge_rss
la riga live:
    [rss]  code attive: porta1=14/16(min=812,max=1390) ...
Se il numero resta a 1/16 (o comunque basso) con traffico in corso, l'hash
non sta distribuendo — le RXQ DOCA GPUNetIO sono code raw DevX, NON visibili
via "ethtool -S": questo contatore è l'UNICO modo per osservarle dall'esterno.
Ogni ~5 tick (~5s) viene anche stampato il conteggio per singola coda
("porta1: [0]=... [1]=... ..."), utile per un istogramma più fine.

Tunable disponibili (via -t key=value o --prof-tun):
    pkt_size    (int)  default 64     - dimensione L2 in byte (vedi nota FCS
                                         in gpu_bridge_stress.py: per un frame
                                         RFC2544 di N byte sul cavo passa
                                         pkt_size=N-4)
    n_src_ip    (int)  default 4096   - quanti indirizzi src IP distinti
                                         (10.0.0.1 .. 10.0.0.1+n_src_ip-1)
    n_dst_ip    (int)  default 1      - quanti indirizzi dst IP distinti
                                         (1 = fisso "20.0.0.1", come negli
                                         altri profili; >1 per variare anche
                                         il dst)
    n_sport     (int)  default 60000  - quante porte UDP sorgente distinte
                                         (1024 .. 1024+n_sport-1)
    cache_size  (int)  default 0      - 0 = nessuna cache, rigenera random ad
                                         ogni pacchetto (massima entropia,
                                         più costoso per la CPU di T-Rex ad
                                         alto pps); >0 = pre-genera N varianti
                                         e ricicla (più leggero, copre meno
                                         hash-space — usa un multiplo di 16
                                         per non "sfortunatamente" allinearti
                                         male con il numero di code)

Uso manuale da trex-console (PROVA QUESTO PER PRIMO):
    trex> start -f gpu_bridge_rss_stress.py -m 100kpps -p 1

Con più entropia / rate più alto:
    trex> start -f gpu_bridge_rss_stress.py -m 2mpps -p 1 \\
        -t n_src_ip=16384,n_sport=60000,cache_size=0

Ferma il traffico con: stop -p 1
Statistiche live: stats -p 1  oppure  stats -a  oppure  tui
"""
from trex_stl_lib.api import *


def _ip_add(ip, n):
    """Somma n a un indirizzo IPv4 dotted-string (rimane nello stesso /8
    per i range usati in questo profilo, nessuna gestione overflow oltre)."""
    a, b, c, d = (int(x) for x in ip.split('.'))
    val = ((a << 24) | (b << 16) | (c << 8) | d) + n
    return "%d.%d.%d.%d" % ((val >> 24) & 0xFF, (val >> 16) & 0xFF,
                             (val >> 8) & 0xFF, val & 0xFF)


class STLS1(object):

    def create_stream(self, pkt_size=64, n_src_ip=4096, n_dst_ip=1,
                       n_sport=60000, cache_size=0):
        base_src_ip = "10.0.0.1"
        base_dst_ip = "20.0.0.1"

        base_pkt = Ether() / IP(src=base_src_ip, dst=base_dst_ip) / UDP(dport=2090, sport=1234)
        pad_len = max(0, pkt_size - len(base_pkt))

        instr = [
            STLVmFlowVar(name="ip_src", min_value=base_src_ip,
                         max_value=_ip_add(base_src_ip, max(0, n_src_ip - 1)),
                         size=4, op="random"),
            STLVmWrFlowVar(fv_name="ip_src", pkt_offset="IP.src"),
        ]

        if n_dst_ip > 1:
            instr += [
                STLVmFlowVar(name="ip_dst", min_value=base_dst_ip,
                             max_value=_ip_add(base_dst_ip, n_dst_ip - 1),
                             size=4, op="random"),
                STLVmWrFlowVar(fv_name="ip_dst", pkt_offset="IP.dst"),
            ]

        instr += [
            STLVmFlowVar(name="sport", min_value=1024,
                         max_value=1024 + max(0, n_sport - 1),
                         size=2, op="random"),
            STLVmWrFlowVar(fv_name="sport", pkt_offset="UDP.sport"),
            # Checksum IP ricalcolato dopo la scrittura degli indirizzi.
            # Il checksum UDP resta 0 (valido per RFC 768, "nessun checksum"):
            # dato che non tocchiamo il payload la cosa non crea pacchetti
            # malformati, solo non validati lato UDP.
            STLVmFixIpv4(offset="IP"),
        ]

        vm = STLScVmRaw(instr, cache_size=cache_size)

        pkt = STLPktBuilder(pkt=base_pkt / (b'\x00' * pad_len), vm=vm)

        return STLStream(packet=pkt, mode=STLTXCont(pps=1000))

    def get_streams(self, direction=0, port_id=0, pkt_size=64, n_src_ip=4096,
                     n_dst_ip=1, n_sport=60000, cache_size=0, **kwargs):
        return [self.create_stream(pkt_size=pkt_size, n_src_ip=n_src_ip,
                                    n_dst_ip=n_dst_ip, n_sport=n_sport,
                                    cache_size=cache_size)]


# entry point richiesto da T-Rex
def register():
    return STLS1()
