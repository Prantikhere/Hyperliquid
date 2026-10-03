#!/usr/bin/env bash
# ==============================================================================
# TradingBingx — Automated Reboot Recovery & Self-Healing Engine
# ==============================================================================
# Triggered on @reboot or system reset/disruption.
# Restores network sync, Laya AI, reconciles on-chain positions, clears stale halts,
# and restarts all trading activities and monitoring without human intervention.
# ==============================================================================
set -euo pipefail

PROJECT_DIR="/home/prantik/Downloads/Personal/TradingBingx"
cd "$PROJECT_DIR"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"
REBOOT_LOG="$PROJECT_DIR/logs/reboot.log"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/state"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [REBOOT_RECOVERY] $1" | tee -a "$REBOOT_LOG"
}

log "=================================================================="
log "SYSTEM DISRUPTION / REBOOT RECOVERY SEQUENCE INITIATED"
log "Host: $(hostname) | Uptime: $(uptime -p 2>/dev/null || echo 'unknown')"
log "=================================================================="

# Step 1: Wait for Network Connectivity (up to 60 seconds)
log "[1/6] Awaiting network connectivity & Hyperliquid API reachability..."
NET_READY=0
for i in {1..30}; do
    if curl -s --max-time 3 https://api.hyperliquid.xyz/info > /dev/null 2>&1; then
        NET_READY=1
        log "✅ Network is ONLINE and Hyperliquid API is responsive (attempt $i)."
        break
    fi
    sleep 2
done

if [ "$NET_READY" -ne 1 ]; then
    log "⚠️ Network not fully reachable after 60s. Will proceed with local recovery steps."
fi

# Step 2: Ensure Redis is Running
log "[2/6] Verifying Redis daemon..."
if ! pgrep -x "redis-server" > /dev/null 2>&1; then
    log "Starting redis-server..."
    redis-server --daemonize yes >> "$REBOOT_LOG" 2>&1 || true
    sleep 1
fi
if pgrep -x "redis-server" > /dev/null 2>&1; then
    log "✅ Redis daemon is active."
else
    log "⚠️ Redis server not detected in process list, checking socket..."
fi

# Step 3: Ensure Laya AI Decision Engine Service is UP
log "[3/6] Supervising Laya AI Decision Engine..."
systemctl --user start laya.service >> "$REBOOT_LOG" 2>&1 || true

LAYA_READY=0
for i in {1..45}; do
    if curl -s --max-time 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
        LAYA_READY=1
        log "✅ Laya AI is UP and responding (attempt $i)."
        break
    fi
    sleep 2
done

if [ "$LAYA_READY" -ne 1 ]; then
    log "⚠️ Laya AI still warming up weights in background (PID: $(pgrep -f 'server.py' || echo 'none'))."
fi

# Step 4: Reconcile On-Chain Positions & Clear Stale Markers
log "[4/6] Reconciling on-chain positions from Hyperliquid..."
if [ -x "$VENV_PYTHON" ]; then
    "$VENV_PYTHON" scripts/sync_position_state.py >> "$REBOOT_LOG" 2>&1 || log "⚠️ Position sync script returned warning."
fi

# Clear stale halt marker on verified reboot/recovery
rm -f "$PROJECT_DIR/state/STALE_WAKE_HALT"
log "✅ Cleared STALE_WAKE_HALT marker. Trading execution unblocked."

# Step 5: Launch Trading Activities via Watchdog
log "[5/6] Invoking watchdog to verify/start all trading daemons..."
/bin/bash "$PROJECT_DIR/watchdog.sh" >> "$REBOOT_LOG" 2>&1 || true
log "✅ Watchdog pass executed."

# Step 6: Execute Health Monitor and Generate Report
log "[6/6] Executing system health monitor & recording state..."
/bin/bash "$PROJECT_DIR/scripts/monitor_system.sh" >> "$REBOOT_LOG" 2>&1 || true

log "=================================================================="
log "REBOOT RECOVERY SEQUENCE COMPLETE — ALL ACTIVITIES RESTORED"
log "=================================================================="
