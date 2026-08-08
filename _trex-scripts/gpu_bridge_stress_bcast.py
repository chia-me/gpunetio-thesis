"""
Profilo stateless T-Rex per stress-test del GPU packet forwarder (BF2 + A30).
Obiettivo: massimizzare il PACKET RATE (pps), non la dimensione del pacchetto,
per capire se/come il forwarder crea un CUDA kernel per pacchetto e quanti
riesce a gestire prima di saturare o perdere pacchetti.

Topologia (porte fisiche confermate da /etc/trex_cfg.yaml):
    interfaces: ["2a:00.0", "2c:00.0"] -> porta logica 0 = 2a:00.0, porta logica 1 = 2c:00.0

    gpu_bridge e' simmetrico (puo' forwardare in entrambe le direzioni), ma
    questo profilo e' pensato per un test volutamente unidirezionale:
    T-Rex trasmette verso ad:00.1 (ingress), il traffico esce da ad:00.0.
    Dato che 2c:00.0 (porta 1) e' sulla stessa VLAN di ad:00.1, il TX va
    fatto da porta 1 (-p 1); la porta 0 riceve il ritorno da ad:00.0.
    dest_mac = broadcast, l'unica configurazione finora confermata funzionante.

Tunable disponibili (via --prof-tun key=value, es: --prof-tun pkt_size=256):
    pkt_size          (int)  default 64    - dimensione pacchetto L2 in byte
                                              costruita in software (Scapy).
                                              Il FCS (4 byte di CRC) viene
                                              aggiunto dopo dall'hardware della
                                              NIC e non è incluso in questo
                                              numero: sul cavo il frame reale è
                                              pkt_size+4 byte. Per ottenere una
                                              taglia RFC 2544 esatta (dove 64,
                                              128, ... includono il FCS), passa
                                              pkt_size = taglia_RFC - 4, es.
                                              pkt_size=60 per un frame da 64B
                                              sul cavo.
    enable_flow_stats (bool) default False - se True aggiunge STLFlowLatencyStats
                                              per tracciare drop/dup/out_of_order
                                              su questo flusso (serve per il test
                                              di riordinamento)
    pg_id             (int)  default 1     - packet group id per flow stats

Uso manuale da trex-console (usa i default, nessun tunable) - PROVA QUESTO PER PRIMO:
    trex> start -f gpu_bridge_stress.py -m 100kpps -p 1

Uso manuale con tunable da console:
    trex> start -f gpu_bridge_stress.py -m <rate> -p 1 -t pkt_size=256

Uso con ./ndr (via --prof-tun), una volta confermato che il test manuale funziona:
    ./ndr --stl --ports 1 0 --profile gpu_bridge_stress.py \\
        --prof-tun pkt_size=256 -p 0.1 -e 1 -q 2 -t 20 -ft 20 -x 12 \\
        --opt-bin-search -v -o hu

NOTA SULLE PORTE: per ./ndr, verifica quale ordine tra "--ports 1 0" e
"--ports 0 1" viene accettato da stl_map_ports (potrebbe essere cambiato
dall'ultima verifica) - quello che conta e' che il TX finisca sulla porta 1.

Ferma il traffico con: stop -p 1
Statistiche live: stats -p 1  oppure  stats -a  oppure  tui
"""
from trex_stl_lib.api import *

# NOTA: broadcast come dest_mac e' l'UNICA configurazione empiricamente
# verificata funzionare con questo bridge (NDR di riferimento ~460Mbps).
# Tentativi con MAC unicast (sia quello del bridge, sia quello dell'altra
# porta T-Rex) non hanno funzionato - probabile che gpu_bridge non sia un
# bridge simmetrico con apprendimento bidirezionale ma un forwarder a
# verso fisso (ingress sempre ad:00.1, egress sempre ad:00.0, per come
# viene lanciato con -n ad:00.0 -n ad:00.1 -g b). Broadcast bypassa il
# problema perche' non richiede che nulla sia "imparato" da nessuno.
DEST_MAC = "ff:ff:ff:ff:ff:ff"


class STLS1(object):

    def create_stream(self, port_id=0, pkt_size=64, enable_flow_stats=False, pg_id=1):
        base_pkt = Ether(dst=DEST_MAC) / IP(src="16.0.0.1", dst="10.0.0.1") / UDP(dport=2090, sport=1234)

        pad_len = max(0, pkt_size - len(base_pkt))
        pkt = STLPktBuilder(pkt=base_pkt / (b'\x00' * pad_len))

        flow_stats = STLFlowLatencyStats(pg_id=pg_id) if enable_flow_stats else None

        return STLStream(
            packet=pkt,
            mode=STLTXCont(pps=1000),  # rate di base, sovrascritto da -m in CLI o da ./ndr
            flow_stats=flow_stats,
        )

    def get_streams(self, direction=0, port_id=0, pkt_size=64,
                     enable_flow_stats=False, pg_id=1, **kwargs):
        return [self.create_stream(port_id=port_id, pkt_size=pkt_size,
                                    enable_flow_stats=enable_flow_stats, pg_id=pg_id)]


# entry point richiesto da T-Rex
def register():
    return STLS1()
