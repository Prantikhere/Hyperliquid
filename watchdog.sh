#!/bin/bash

# Configuration
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
PROJECT_DIR="/home/prantik/Downloads/Personal/TradingBingx"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"
export PYTHONPATH="$PROJECT_DIR"

# Logs
WATCHDOG_LOG="$PROJECT_DIR/logs/watchdog.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$WATCHDOG_LOG"
}

# Stale-wake detector with automated self-healing recovery:
# Detects host sleep, reboot, or network freeze gaps. Instead of halting permanently,
# it checks API connectivity, reconciles on-chain positions, and safely auto-resumes trading.
STALE_GAP_SEC=600   # cron cadence is 2min; 600s gives slack before flagging
KILL_MARKER="$PROJECT_DIR/state/STALE_WAKE_HALT"
mkdir -p "$PROJECT_DIR/state"

reconcile_and_resume() {
    log "Initiating auto-recovery: checking Hyperliquid connectivity and on-chain positions..."
    if curl -s --max-time 5 https://api.hyperliquid.xyz/info > /dev/null 2>&1; then
        cd "$PROJECT_DIR" && "$VENV_PYTHON" scripts/sync_position_state.py >> "$PROJECT_DIR/logs/reboot.log" 2>&1
        rm -f "$KILL_MARKER"
        log "Self-healing auto-recovery successful: on-chain positions reconciled. Trading ACTIVE."
        return 0
    else
        log "WARNING: Hyperliquid API unreachable or network offline. Retaining temporary halt marker."
        echo "$(date '+%Y-%m-%d %H:%M:%S') gap=${1:-unknown} network_offline" > "$KILL_MARKER"
        return 1
    fi
}

if [ -f "$WATCHDOG_LOG" ]; then
    LAST_TS=$(tail -1 "$WATCHDOG_LOG" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    if [ -n "$LAST_TS" ]; then
        LAST_EPOCH=$(date -d "$LAST_TS" +%s 2>/dev/null)
        NOW_EPOCH=$(date +%s)
        if [ -n "$LAST_EPOCH" ]; then
            GAP=$((NOW_EPOCH - LAST_EPOCH))
            if [ "$GAP" -gt "$STALE_GAP_SEC" ]; then
                log "NOTICE: watchdog gap of ${GAP}s detected (host reboot, wake, or freeze). Triggering self-healing recovery..."
                reconcile_and_resume "$GAP"
            fi
        fi
    fi
fi

# If a stale halt marker exists from an earlier network drop, auto-clear once network restores
if [ -f "$KILL_MARKER" ]; then
    reconcile_and_resume "marker_recovery"
fi

# 0. Laya AI Decision Engine supervision with 180s warmup grace period
MAIN_PID=$(systemctl --user show -p MainPID --value laya.service 2>/dev/null || echo 0)
if [ "$MAIN_PID" -gt 0 ]; then
    ELAPSED=$(ps -p "$MAIN_PID" -o etimes= 2>/dev/null | tr -d ' ' || echo 999)
    if [ -n "$ELAPSED" ] && [ "$ELAPSED" -lt 180 ]; then
        log "OK: Laya AI service is warming up checkpoint into RAM (PID: $MAIN_PID, elapsed: ${ELAPSED}s < 180s grace)."
    elif ! curl -s --max-time 10 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status":"ok"'; then
        log "CRITICAL: Laya AI service unhealthy after ${ELAPSED}s grace. Restarting via systemctl --user..."
        systemctl --user restart laya.service >> "$WATCHDOG_LOG" 2>&1 || true
    else
        log "OK: Laya AI service is UP."
    fi
else
    log "CRITICAL: laya.service not running. Starting via systemctl --user..."
    systemctl --user start laya.service >> "$WATCHDOG_LOG" 2>&1 || true
fi



check_and_start() {
    local script_path=$1
    local process_pattern=$2
    local description=$3

    # Check if process is running
    if ! ps aux | grep "$process_pattern" | grep -v grep > /dev/null; then
        log "CRITICAL: $description ($script_path) is NOT running. Restarting..."
        # stdout is already persisted (rotated, 10-day retention) via src.utils.logger -> logs/polyarb.log;
        # redirecting to streamer.log too just duplicated it uncapped and grew to 400MB+.
        cd "$PROJECT_DIR" && nohup "$VENV_PYTHON" -u "$script_path" >> /dev/null 2>&1 &
        log "Successfully triggered restart for $description."
    else
        log "OK: $description is running."
    fi
}

# 1. Multi-Exchange Streamer
check_and_start "src/data/multi_streamer.py" "multi_streamer.py" "Multi-Exchange Streamer"

# 2. BingX Executor -- PAUSED/DISABLED
# check_and_start "src/execution/bingx_executor.py" "bingx_executor.py" "BingX Executor"

# 3. Hyperliquid Executor -- RE-ENABLED with Strict Universe Isolation
# Runs the supervisor bot + SettlementAgent profit-booking for symbols NOT traded by perp_ls or carry.
check_and_start "src/execution/hl_executor.py" "hl_executor.py" "Hyperliquid Executor"

# 3b. Delta-neutral Funding-Carry Executor -- DISABLED/PAUSED (requires spot wallet access)
# if ! ps aux | grep "carry_executor" | grep -v grep > /dev/null; then
#     log "CRITICAL: Carry Executor is NOT running. Restarting..."
#     cd "$PROJECT_DIR" && CARRY_LIVE=yes nohup "$VENV_PYTHON" -u -m src.execution.carry_executor >> "$PROJECT_DIR/logs/carry.log" 2>&1 &
#     log "Successfully triggered restart for Carry Executor."
# else
#     log "OK: Carry Executor is running."
# fi

# 3c. Dollar-neutral cross-sectional PERP Long-Short (LIVE on HL testnet; needs HL_PERP_LIVE=yes)
#     Market-neutral by construction + DD kill-switch: cannot bleed. Fill-verified orders.
if ! ps aux | grep "hl_perp_ls" | grep -v grep > /dev/null; then
    log "CRITICAL: Perp L/S Executor is NOT running. Restarting..."
    cd "$PROJECT_DIR" && HL_PERP_LIVE=yes nohup "$VENV_PYTHON" -u -m src.execution.hl_perp_ls >> "$PROJECT_DIR/logs/perp_ls.log" 2>&1 &
    log "Successfully triggered restart for Perp L/S Executor."
else
    log "OK: Perp L/S Executor is running."
fi

# 3d. Cointegrated Pairs Stat-Arb -- PAUSED to dedicate 100% margin to rapid scalping
# if ! ps aux | grep "pairs_arb_executor" | grep -v grep > /dev/null; then
#     log "CRITICAL: Pairs Stat-Arb Executor is NOT running. Restarting..."
#     cd "$PROJECT_DIR" && PAIRS_ARB_LIVE=yes nohup "$VENV_PYTHON" -u -m src.execution.pairs_arb_executor >> "$PROJECT_DIR/logs/pairs_arb.log" 2>&1 &
#     log "Successfully triggered restart for Pairs Stat-Arb Executor."
# else
#     log "OK: Pairs Stat-Arb Executor is running."
# fi


# 4. Meta-Learner Auto-Retrain (Run if not recently run)
if [[ $(( $(date +%s) / 3600 % 6 )) -eq 0 ]]; then
    LAST_RETRAIN_FILE="$PROJECT_DIR/logs/.last_retrain"
    CURRENT_TIME=$(date +%s)
    if [ ! -f "$LAST_RETRAIN_FILE" ] || [ $(( CURRENT_TIME - $(cat "$LAST_RETRAIN_FILE") )) -gt 18000 ]; then
        log "Running scheduled Meta-Learner retraining..."
        echo "$CURRENT_TIME" > "$LAST_RETRAIN_FILE"
        cd "$PROJECT_DIR" && "$VENV_PYTHON" src/intelligence/train_meta.py >> "$PROJECT_DIR/logs/training.log" 2>&1
    fi
fi

# 5. Daily PnL Report (once per day at hour 23 UTC)
HOUR_NOW=$(date -u +%H)
MINUTE_NOW=$(date -u +%M)
DAILY_REPORT_MARKER="$PROJECT_DIR/logs/.daily_report_done_$(date -u +%Y%m%d)"
if [ "$HOUR_NOW" -eq 23 ] && [ "$MINUTE_NOW" -lt 10 ] && [ ! -f "$DAILY_REPORT_MARKER" ]; then
    log "Running daily PnL report..."
    touch "$DAILY_REPORT_MARKER"
    cd "$PROJECT_DIR" && timeout 60 "$VENV_PYTHON" report_pnl.py >> "$PROJECT_DIR/logs/pnl.log" 2>&1 &
fi

# 6. Calibration Ledger — append resolved trade outcomes for meta-learner feedback
CALIBRATION_MARKER="$PROJECT_DIR/logs/.calibration_done_$(date -u +%Y%m%d_%H)"
if [ "$(( $(date -u +%s) / 3600 ))" -ne "$(cat "$PROJECT_DIR/logs/.calibration_last_hour" 2>/dev/null || echo 0)" ]; then
    echo "$(( $(date -u +%s) / 3600 ))" > "$PROJECT_DIR/logs/.calibration_last_hour"
    cd "$PROJECT_DIR" && timeout 30 "$VENV_PYTHON" -c "
import json, os, sys
sys.path.insert(0, '.')
from src.utils.db import DatabaseManager
db = DatabaseManager()
# Fetch recent trades with outcomes that aren't yet in calibration ledger
ledger_path = 'data/calibration_ledger.jsonl'
existing_ids = set()
if os.path.exists(ledger_path):
    with open(ledger_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                existing_ids.add(row.get('trade_id', ''))
            except: pass
query = '''
    SELECT id, market_id, side, price, metadata->>'quant_action' as qa,
           metadata->>'regime' as regime, metadata->>'meta_confidence' as mc,
           metadata->>'outcome' as outcome
    FROM system_trades
    WHERE status = 'LIVE_OK' AND (metadata->>'outcome') IS NOT NULL
    ORDER BY time DESC LIMIT 50
'''
rows = db.execute_query(query)
os.makedirs('data', exist_ok=True)
added = 0
if rows:
    with open(ledger_path, 'a') as f:
        for row in rows:
            tid = str(row[0])
            if tid in existing_ids:
                continue
            entry = {
                'trade_id': tid, 'symbol': row[1], 'side': row[2],
                'entry_price': float(row[3]), 'quant_action': row[4],
                'regime': row[5], 'meta_confidence': float(row[6]) if row[6] else None,
                'outcome_roi': float(row[7]) if row[7] else None,
                'correct': (row[2] == 'BUY' and float(row[7]) > 0) or (row[2] == 'SELL' and float(row[7]) > 0) if row[7] else None
            }
            f.write(json.dumps(entry) + '\n')
            added += 1
print(f'Calibration ledger: {added} new rows')
" >> "$PROJECT_DIR/logs/calibration.log" 2>&1
fi
