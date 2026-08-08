"""
Profilo stateless T-Rex per stress-test del GPU packet forwarder (BF2 + A30).

Obiettivo: massimizzare il PACKET RATE (pps), non la dimensione del pacchetto,
per capire se/come il forwarder crea un CUDA kernel per pacchetto e quanti
riesce a gestire prima di saturare o perdere pacchetti.

Uso:
    cd /opt/trex/v3.08
    ./trex-console
    trex> start -f gpu_forwarder_stress.py -m <rate> -p 1

Esempi di sweep (lanciali uno alla volta, guardando ipackets/opackets/drop-rate
sul terminale server e i contatori CUDA lato gpu_forwarder):
    start -f gpu_forwarder_stress.py -m 100kpps -p 1
    start -f gpu_forwarder_stress.py -m 1mpps  -p 1
    start -f gpu_forwarder_stress.py -m 5mpps  -p 1
    start -f gpu_forwarder_stress.py -m 10mpps -p 1

Nota: -p 1 è OBBLIGATORIO -> è la porta reale (2c:00.0) collegata allo switch.
La porta 0 (2a:00.0) è dummy, non collegata a nulla.

Ferma il traffico con: stop -p 1
Statistiche live: stats -p 1  oppure  tui
"""

from trex_stl_lib.api import *


class STLS1(object):
    def __init__(self):
        # Dimensione pacchetto minima per non essere bandwidth-bound:
        # 64 byte L2 (minimo Ethernet) cosi il collo di bottiglia e' il pps,
        # non i Gbps. Non importa il contenuto, solo il rate di generazione.
        self.pkt_len = 64

    def create_stream(self):
        # Payload minimo per arrivare a 64 byte totali (Ether+IP+UDP+pad)
        base_pkt = Ether() / IP(src="16.0.0.1", dst="10.0.0.1") / UDP(dport=2090, sport=1234)
        pad_len = max(0, self.pkt_len - len(base_pkt))
        pkt = STLPktBuilder(pkt=base_pkt / (b'\x00' * pad_len))

        return STLStream(
            packet=pkt,
            mode=STLTXCont(pps=1000),  # rate di base, sovrascritto da -m in CLI
        )

    def get_streams(self, direction=0, **kwargs):
        # direction e' ignorato: usiamo esplicitamente -p 1 da console,
        # cosi il traffico va sempre e solo sulla porta reale verso la BF2.
        return [self.create_stream()]


# entry point richiesto da T-Rex
def register():
    return STLS1()
