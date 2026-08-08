#!/bin/bash
#
# wd_bridge.sh   (ex gpu_bridge_watchdog.sh)
#
# NOTA SUL NOME: questo script NON deve contenere la stringa "gpu_bridge"
# nel proprio nome file ne' nel tag di sessione. Un "pkill -f gpu_bridge"
# (tuo, o di un altro script) colpirebbe anche questo processo.
# Quindi: NON usare tag tipo "gpu_bridge_test" come argomento.
#
# Lancia gpu_bridge dentro il netns bf2 e ne monitora il funzionamento.
#
# COSA SORVEGLIA:
#   Il guasto osservato non e' un blocco della RICEZIONE (il bridge continua
#   a ricevere, contare e stampare) ma la morte silenziosa della
#   TRASMISSIONE. Quindi la condizione di freeze e':
#
#       la NIC di INGRESSO riceve traffico vero (> MIN_NIC_DELTA pkt/s)
#       MA la NIC di USCITA non trasmette nulla
#
#   Entrambi i contatori sono hardware (ethtool), indipendenti da cosa
#   stampa il bridge: nessun problema di buffering o di contatori interni.
#
# USO:
#   sudo ./wd_bridge.sh [nome_prova]
#   AUTO_RESTART=0 sudo -E ./wd_bridge.sh nome_prova    # solo rileva, non riavvia
#
# RIAVVIO SU RICHIESTA (per automatizzare uno sweep):
#   Da un altro terminale / script:
#       touch /tmp/bridge_restart_req
#   Il watchdog riavvia gpu_bridge e cancella il file. Utile per partire
#   da un bridge pulito prima di ogni corsa ./ndr.
#
# Ogni evento (freeze, riavvio, crash) e' registrato con timestamp in
# LOG_DIR/<nome_prova>_watchdog_events.log: incrocia quei timestamp con i
# log di ./ndr per scartare le misure contaminate.

set -u

# ── Sessione e cartella log ──────────────────────────────────────────────
SESSION_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="/home/prognose/gpu_bridge_logs"
mkdir -p "$LOG_DIR"

# ── Configurazione ───────────────────────────────────────────────────────
NETNS="bf2"
BRIDGE_DIR="/home/prognose/doca-gpunetio/prog_gpu_bridge"
BRIDGE_BIN="./gpu_bridge"
BRIDGE_ARGS="-n ad:00.0 -n ad:00.1 -g b0:00.0"

LOGFILE="${LOG_DIR}/${SESSION_TAG}_bridge_live.log"
EVENTFILE="${LOG_DIR}/${SESSION_TAG}_watchdog_events.log"

# Interfaccia di INGRESSO (ad:00.1): dove T-Rex trasmette.
NIC_IN_IF="enp173s0f1np1"
NIC_RX_FIELD="rx_packets_phy"

# Interfaccia di USCITA (ad:00.0): da dove il bridge deve rimandare fuori.
# VERIFICA questo nome prima di fidarti:
#   sudo ip netns exec bf2 ip -br link
NIC_OUT_IF="enp173s0f0np0"
NIC_TX_FIELD="tx_packets_phy"

# File trigger per il riavvio su richiesta.
TRIGGER_FILE="/tmp/bridge_restart_req"

# ── Soglie ───────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC=1

# Secondi consecutivi di divergenza (RX sale, TX fermo) prima del verdetto.
STALL_THRESHOLD_SEC=10

# Pacchetti/s minimi in ingresso perche' la situazione sia valutabile.
# Sotto questa soglia c'e' solo traffico di controllo dello switch
# (LLDP, STP, multicast) e il silenzio in TX non significa nulla.
MIN_NIC_DELTA=20000

WARMUP_SEC=15        # init DOCA/GPU dopo l'avvio
COOLDOWN_SEC=20      # pausa dopo un riavvio prima di rivalutare
SIGTERM_GRACE_SEC=5
HEARTBEAT_INTERVAL_SEC=10

AUTO_RESTART="${AUTO_RESTART:-1}"

# ─────────────────────────────────────────────────────────────────────────

BRIDGE_PID=""

log_event() {
    echo "[watchdog] $(date '+%H:%M:%S') $*"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$EVENTFILE"
}

start_bridge() {
    log_event "avvio gpu_bridge"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') avvio (tag: ${SESSION_TAG}) =====" >> "$LOGFILE"

    # exec: la subshell viene SOSTITUITA dal processo, cosi' $! e' il PID
    #       reale di gpu_bridge (senza, sarebbe il PID della subshell e il
    #       kill non arriverebbe al figlio, lasciandolo orfano).
    # stdbuf -oL: stdout line-buffered anche scrivendo su file.
    (
        cd "$BRIDGE_DIR" || exit 1
        exec stdbuf -oL -eL ip netns exec "$NETNS" $BRIDGE_BIN $BRIDGE_ARGS
    ) >> "$LOGFILE" 2>&1 &

    BRIDGE_PID=$!
    log_event "gpu_bridge PID=$BRIDGE_PID"
}

stop_bridge() {
    [ -n "$BRIDGE_PID" ] || return 0
    if kill -0 "$BRIDGE_PID" 2>/dev/null; then
        log_event "SIGTERM a PID=$BRIDGE_PID"
        kill -TERM "$BRIDGE_PID" 2>/dev/null
        for _ in $(seq 1 "$SIGTERM_GRACE_SEC"); do
            kill -0 "$BRIDGE_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$BRIDGE_PID" 2>/dev/null; then
            log_event "ancora vivo dopo SIGTERM, SIGKILL"
            kill -KILL "$BRIDGE_PID" 2>/dev/null
        fi
        wait "$BRIDGE_PID" 2>/dev/null
    fi
    BRIDGE_PID=""
    # NESSUN pkill qui: con exec il PID e' quello giusto e kill basta.
    # Il pkill -f su "gpu_bridge" colpiva anche questo script -> loop infinito.
    # Il controllo orfani si fa a mano prima di lanciare: pgrep -af gpu_bridge
}

restart_bridge() {
    stop_bridge
    start_bridge
    sleep "$COOLDOWN_SEC"
    stall_count=0
    last_rx=""
    last_tx=""
}

cleanup_and_exit() {
    trap - SIGINT SIGTERM      # disarma subito: un secondo Ctrl-C non rientra
    log_event "uscita richiesta, fermo gpu_bridge"
    stop_bridge
    exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

if [ "$(id -u)" -ne 0 ]; then
    echo "Serve root (ip netns exec). Rilancia con sudo."
    exit 1
fi

# Guardia contro il problema del nome: se il tag contiene "gpu_bridge",
# un pkill esterno potrebbe colpire questo processo.
case "$SESSION_TAG" in
    *gpu_bridge*)
        echo "ERRORE: il tag di sessione non deve contenere 'gpu_bridge'."
        echo "Usa un nome diverso, es: sweep_rfc2544"
        exit 1
        ;;
esac

get_counter() {
    # $1 = interfaccia, $2 = nome esatto del campo ethtool
    ip netns exec "$NETNS" ethtool -S "$1" 2>/dev/null \
        | awk -F: -v f="$2" \
              '{gsub(/^[ \t]+|[ \t]+$/,"",$1)} $1==f {gsub(/[^0-9]/,"",$2); print $2; exit}'
}

: > "$LOGFILE"
: > "$EVENTFILE"
rm -f "$TRIGGER_FILE"

echo "[watchdog] prova          : ${SESSION_TAG}"
echo "[watchdog] log bridge     : ${LOGFILE}"
echo "[watchdog] log eventi     : ${EVENTFILE}"
echo "[watchdog] auto-restart   : ${AUTO_RESTART}"
echo "[watchdog] trigger riavvio: touch ${TRIGGER_FILE}"

# Verifica i due contatori PRIMA di partire: se un nome e' sbagliato il
# watchdog non puo' funzionare, meglio saperlo subito.
probe_rx="$(get_counter "$NIC_IN_IF" "$NIC_RX_FIELD")"
if [ -z "$probe_rx" ]; then
    echo "ERRORE: campo '$NIC_RX_FIELD' non trovato su $NIC_IN_IF"
    echo "Campi disponibili:"
    ip netns exec "$NETNS" ethtool -S "$NIC_IN_IF" 2>/dev/null | grep -i 'rx.*packet'
    exit 1
fi
probe_tx="$(get_counter "$NIC_OUT_IF" "$NIC_TX_FIELD")"
if [ -z "$probe_tx" ]; then
    echo "ERRORE: campo '$NIC_TX_FIELD' non trovato su $NIC_OUT_IF"
    echo "Interfacce nel netns ${NETNS}:"
    ip netns exec "$NETNS" ip -br link
    exit 1
fi
echo "[watchdog] contatori ok: ${NIC_IN_IF}/${NIC_RX_FIELD}=${probe_rx}  ${NIC_OUT_IF}/${NIC_TX_FIELD}=${probe_tx}"

# Avvisa se c'e' gia' un gpu_bridge in giro: due istanze sulle stesse code
# rompono tutto, ed e' meglio deciderlo a mano che ammazzarlo di nascosto.
if pgrep -f -- "$BRIDGE_BIN $BRIDGE_ARGS" >/dev/null 2>&1; then
    echo
    echo "ATTENZIONE: risulta gia' in esecuzione un gpu_bridge:"
    pgrep -af -- "$BRIDGE_BIN $BRIDGE_ARGS"
    echo "Due istanze si contendono le stesse code della NIC."
    echo "Terminalo a mano e rilancia questo script."
    exit 1
fi

start_bridge

echo "[watchdog] warmup ${WARMUP_SEC}s (init DOCA/GPU)..."
sleep "$WARMUP_SEC"

last_rx=""
last_tx=""
stall_count=0
heartbeat_count=0

echo "[watchdog] monitoraggio avviato"
echo "[watchdog] freeze = RX in > ${MIN_NIC_DELTA} pkt/s MA TX out fermo, per ${STALL_THRESHOLD_SEC}s"

while true; do
    sleep "$POLL_INTERVAL_SEC"

    # ── riavvio su richiesta esterna ─────────────────────────────────────
    if [ -f "$TRIGGER_FILE" ]; then
        rm -f "$TRIGGER_FILE"
        log_event "RIAVVIO su richiesta esterna (trigger file)"
        restart_bridge
        continue
    fi

    # ── il processo e' morto da solo? ────────────────────────────────────
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        log_event "CRASH: gpu_bridge uscito da solo"
        if [ "$AUTO_RESTART" = "1" ]; then
            restart_bridge
        else
            log_event "AUTO_RESTART=0: non riavvio, esco"
            exit 1
        fi
        continue
    fi

    rx=$(get_counter "$NIC_IN_IF" "$NIC_RX_FIELD")
    tx=$(get_counter "$NIC_OUT_IF" "$NIC_TX_FIELD")

    heartbeat_count=$((heartbeat_count + POLL_INTERVAL_SEC))
    if [ "$heartbeat_count" -ge "$HEARTBEAT_INTERVAL_SEC" ]; then
        heartbeat_count=0
        if [ -n "$last_rx" ] && [ -n "$rx" ] && [ -n "$last_tx" ] && [ -n "$tx" ]; then
            echo "[watchdog] $(date '+%H:%M:%S') rx_in=${rx} tx_out=${tx}"
        fi
    fi

    if [ -z "$rx" ] || [ -z "$tx" ]; then
        continue
    fi

    if [ -n "$last_rx" ] && [ -n "$last_tx" ]; then
        rx_delta=$((rx - last_rx))
        tx_delta=$((tx - last_tx))

        # Freeze SOLO se entra traffico vero e non ne esce nulla.
        # delta negativi = contatori azzerati (riavvio): non e' un freeze.
        if [ "$rx_delta" -ge "$MIN_NIC_DELTA" ] && [ "$tx_delta" -le 0 ]; then
            stall_count=$((stall_count + POLL_INTERVAL_SEC))
            echo "[watchdog] $(date '+%H:%M:%S') sospetto ${stall_count}/${STALL_THRESHOLD_SEC}s: rx_in +${rx_delta}, tx_out fermo"
        else
            stall_count=0
        fi
    fi

    last_rx="$rx"
    last_tx="$tx"

    if [ "$stall_count" -ge "$STALL_THRESHOLD_SEC" ]; then
        log_event "FREEZE TX CONFERMATO: il bridge riceve ma non trasmette da ${stall_count}s"
        if [ "$AUTO_RESTART" = "1" ]; then
            log_event "RIAVVIO -- ogni misura ./ndr in corso ORA e' da SCARTARE"
            restart_bridge
        else
            log_event "AUTO_RESTART=0: nessun riavvio, continuo a monitorare"
            stall_count=0
        fi
    fi
done
