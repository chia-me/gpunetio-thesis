#!/bin/bash
#
# gpu_bridge_watchdog.sh
#
# Lancia gpu_bridge dentro il netns bf2 e ne monitora l'output [live].
# Rileva un freeze del kernel persistente e (opzionalmente) riavvia il processo.
#
# USO:
#   sudo ./gpu_bridge_watchdog.sh [nome_prova]
#
#   AUTO_RESTART=0 sudo -E ./gpu_bridge_watchdog.sh prova1
#       -> solo rilevamento e log, nessun riavvio automatico.
#          CONSIGLIATO durante le corse ./ndr: un riavvio a meta' corsa
#          invalida silenziosamente la misura.
#
# Ogni riavvio (o freeze rilevato) viene registrato con timestamp in
# LOG_DIR/<nome_prova>_watchdog_events.log: incrocia quei timestamp con
# i log di ./ndr per scartare i run contaminati.

set -u

# ── Nome della prova e cartella log ──────────────────────────────────────
SESSION_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="/home/prognose/gpu_bridge_logs"
mkdir -p "$LOG_DIR"

# ── Configurazione ───────────────────────────────────────────────────────
NETNS="bf2"
BRIDGE_DIR="/home/prognose/doca-gpunetio/prog_gpu_bridge"
BRIDGE_BIN="./gpu_bridge"
BRIDGE_ARGS="-n ad:00.0 -n ad:00.1 -g b0:00.0"

LOGFILE="${LOG_DIR}/${SESSION_TAG}_gpu_bridge_live.log"
EVENTFILE="${LOG_DIR}/${SESSION_TAG}_watchdog_events.log"

# Interfaccia di INGRESSO del bridge lato BF2 (quella verso cui T-Rex trasmette).
NIC_INGRESS_IF="enp173s0f1np1"
NIC_RX_FIELD="rx_packets_phy"

# ── Soglie ───────────────────────────────────────────────────────────────
POLL_INTERVAL_SEC=1

# Secondi consecutivi di divergenza prima di dichiarare freeze.
# 5s erano troppo pochi: le fasi di ./ndr durano 35s e fra una fase e
# l'altra il traffico si azzera.
STALL_THRESHOLD_SEC=15

# Pacchetti/s minimi visti dalla NIC perche' l'assenza di conteggio nel
# bridge sia considerata sospetta. Serve a ignorare il traffico di
# controllo dello switch (LLDP, STP, multicast) che fa salire comunque
# rx_packets_phy anche quando T-Rex non sta trasmettendo nulla.
MIN_NIC_DELTA=20000

# Attesa dopo l'avvio prima di iniziare a monitorare (init DOCA/GPU).
WARMUP_SEC=15

# Attesa dopo un riavvio prima di ricominciare a valutare i freeze.
COOLDOWN_SEC=20

SIGTERM_GRACE_SEC=5
HEARTBEAT_INTERVAL_SEC=10

# 1 = riavvia automaticamente al freeze; 0 = solo rileva e logga.
AUTO_RESTART="${AUTO_RESTART:-1}"

# ─────────────────────────────────────────────────────────────────────────

BRIDGE_PID=""

log_event() {
    echo "[watchdog] $(date '+%H:%M:%S') $*"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$EVENTFILE"
}

start_bridge() {
    log_event "avvio gpu_bridge"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') avvio gpu_bridge (tag: ${SESSION_TAG}) =====" >> "$LOGFILE"

    # exec: la subshell viene SOSTITUITA dal processo, cosi' $! e' il PID
    # reale di gpu_bridge e non quello di una subshell intermedia.
    # stdbuf -oL: forza stdout line-buffered. Senza questo, scrivendo su
    # file la libc usa blocchi da 4KB e le righe [live] arrivano a raffiche
    # con secondi di ritardo -> falso freeze.
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

    # Rete di sicurezza: se e' rimasto un gpu_bridge orfano (da una sessione
    # precedente o da un kill andato storto) va rimosso, altrimenti al
    # riavvio due processi si contendono le stesse code della NIC.
    if pgrep -f "$(basename "$BRIDGE_BIN")" >/dev/null 2>&1; then
        log_event "trovato gpu_bridge residuo, lo termino"
        pkill -TERM -f "$(basename "$BRIDGE_BIN")" 2>/dev/null
        sleep 2
        pkill -KILL -f "$(basename "$BRIDGE_BIN")" 2>/dev/null
    fi
    BRIDGE_PID=""
}

cleanup_and_exit() {
    log_event "uscita richiesta, fermo gpu_bridge"
    stop_bridge
    exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

if [ "$(id -u)" -ne 0 ]; then
    echo "Serve root (ip netns exec). Rilancia con sudo."
    exit 1
fi

get_nic_rx() {
    # Match esatto sul nome del campo: evita di agganciare per sbaglio
    # rx_prio0_packets_phy o simili a seconda dell'ordine di ethtool.
    ip netns exec "$NETNS" ethtool -S "$NIC_INGRESS_IF" 2>/dev/null \
        | awk -F: -v f="$NIC_RX_FIELD" \
              '{gsub(/^[ \t]+|[ \t]+$/,"",$1)} $1==f {gsub(/[^0-9]/,"",$2); print $2; exit}'
}

get_bridge_rx_tot() {
    # tail prima di grep: il logfile cresce di milioni di righe, non va
    # riletto per intero a ogni giro.
    tail -n 200 "$LOGFILE" 2>/dev/null | grep '^\[live\]' | tail -1 | grep -oP 'rx_tot=\K[0-9]+'
}

: > "$LOGFILE"
: > "$EVENTFILE"
echo "[watchdog] prova: ${SESSION_TAG}"
echo "[watchdog] log gpu_bridge : ${LOGFILE}"
echo "[watchdog] log eventi     : ${EVENTFILE}"
echo "[watchdog] auto-restart   : ${AUTO_RESTART}"

# Verifica il campo ethtool PRIMA di partire: se il nome e' sbagliato il
# watchdog non puo' funzionare e vale la pena saperlo subito.
probe_rx="$(get_nic_rx)"
if [ -z "$probe_rx" ]; then
    echo "ERRORE: campo '$NIC_RX_FIELD' non trovato su $NIC_INGRESS_IF."
    echo "Campi RX disponibili:"
    ip netns exec "$NETNS" ethtool -S "$NIC_INGRESS_IF" 2>/dev/null | grep -i 'rx.*packet'
    exit 1
fi
echo "[watchdog] contatore NIC ok: ${NIC_RX_FIELD}=${probe_rx}"

start_bridge

echo "[watchdog] warmup ${WARMUP_SEC}s (init DOCA/GPU)..."
sleep "$WARMUP_SEC"

last_nic_rx=""
last_bridge_rx=""
stall_count=0
heartbeat_count=0

echo "[watchdog] monitoraggio avviato"
echo "[watchdog] freeze = NIC riceve > ${MIN_NIC_DELTA} pkt/s MA rx_tot del bridge fermo, per ${STALL_THRESHOLD_SEC}s consecutivi"

while true; do
    sleep "$POLL_INTERVAL_SEC"

    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        log_event "CRASH: gpu_bridge uscito da solo"
        if [ "$AUTO_RESTART" = "1" ]; then
            stop_bridge
            start_bridge
            sleep "$COOLDOWN_SEC"
        else
            log_event "AUTO_RESTART=0: non riavvio, esco"
            exit 1
        fi
        stall_count=0; last_nic_rx=""; last_bridge_rx=""
        continue
    fi

    nic_rx=$(get_nic_rx)
    bridge_rx=$(get_bridge_rx_tot)

    heartbeat_count=$((heartbeat_count + POLL_INTERVAL_SEC))
    if [ "$heartbeat_count" -ge "$HEARTBEAT_INTERVAL_SEC" ]; then
        heartbeat_count=0
        if [ -z "$bridge_rx" ]; then
            echo "[watchdog] $(date '+%H:%M:%S') nessuna riga [live] nel log (bridge non ha ancora stampato?)"
        else
            echo "[watchdog] $(date '+%H:%M:%S') NIC rx_phy=${nic_rx}  bridge rx_tot=${bridge_rx}"
        fi
    fi

    if [ -z "$nic_rx" ] || [ -z "$bridge_rx" ]; then
        continue
    fi

    if [ -n "$last_nic_rx" ] && [ -n "$last_bridge_rx" ]; then
        nic_delta=$((nic_rx - last_nic_rx))
        bridge_delta=$((bridge_rx - last_bridge_rx))

        # Freeze SOLO se la NIC sta ricevendo traffico vero (sopra soglia)
        # e il bridge non lo conta affatto. bridge_delta < 0 = contatore
        # ripartito da zero (processo riavviato): non e' un freeze.
        if [ "$nic_delta" -ge "$MIN_NIC_DELTA" ] && [ "$bridge_delta" -eq 0 ]; then
            stall_count=$((stall_count + POLL_INTERVAL_SEC))
            echo "[watchdog] $(date '+%H:%M:%S') sospetto ${stall_count}/${STALL_THRESHOLD_SEC}s: NIC +${nic_delta} pkt, bridge rx_tot fermo a ${bridge_rx}"
        else
            stall_count=0
        fi
    fi

    last_nic_rx="$nic_rx"
    last_bridge_rx="$bridge_rx"

    if [ "$stall_count" -ge "$STALL_THRESHOLD_SEC" ]; then
        log_event "FREEZE CONFERMATO (NIC riceve, bridge fermo da ${stall_count}s)"
        if [ "$AUTO_RESTART" = "1" ]; then
            log_event "RIAVVIO -- ogni misura ./ndr in corso in questo momento e' da SCARTARE"
            stop_bridge
            start_bridge
            sleep "$COOLDOWN_SEC"
        else
            log_event "AUTO_RESTART=0: nessun riavvio, continuo a monitorare"
        fi
        stall_count=0; last_nic_rx=""; last_bridge_rx=""
    fi
done
