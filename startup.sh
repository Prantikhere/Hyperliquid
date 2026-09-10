#!/bin/bash
#==============================================================================
# TradingBingx Auto-Start Script
#==============================================================================
# This script ensures all services start automatically after reboot.
# Add to crontab: @reboot /home/prantik/Downloads/Personal/TradingBingx/startup.sh
#==============================================================================

set -euo pipefail

#--- Configuration ---
PROJECT_DIR="/home/prantik/Downloads/Personal/TradingBingx"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
STATE_DIR="$PROJECT_DIR/state"
PID_DIR="$STATE_DIR/pids"
STARTUP_LOG="$LOG_DIR/startup.log"

#--- Create directories ---
mkdir -p "$LOG_DIR" "$STATE_DIR" "$PID_DIR"

#--- Log function ---
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$STARTUP_LOG"
    echo "$1"
}

#--- Wait for system to stabilize ---
log "System starting up, waiting 30 seconds for network..."
sleep 30

#--- Check if services already running ---
check_running() {
    local pattern=$1
    if ps aux | grep "$pattern" | grep -v grep > /dev/null 2>&1; then
        return 0  # Running
    fi
    return 1  # Not running
}

#--- Start service ---
start_service() {
    local name=$1
    local script=$2
    local pattern=$3
    local log_file=$4
    
    if check_running "$pattern"; then
        log "OK: $name already running"
        return
    fi
    
    log "Starting $name..."
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -u "$script" >> "$log_file" 2>&1 &
    echo $! > "$PID_DIR/$name.pid"
    log "Started $name (PID: $!)"
}

#--- Start watchdog ---
start_watchdog() {
    if check_running "watchdog.sh"; then
        log "OK: Watchdog already running"
        return
    fi
    
    log "Starting Watchdog..."
    cd "$PROJECT_DIR"
    nohup bash watchdog.sh >> /dev/null 2>&1 &
    echo $! > "$PID_DIR/watchdog.pid"
    log "Started Watchdog (PID: $!)"
}

#--- Setup cron if not present ---
setup_cron() {
    if crontab -l 2>/dev/null | grep -q "hourly_monitor.sh"; then
        log "OK: Hourly monitoring cron already active"
    else
        log "Setting up hourly monitoring cron..."
        (crontab -l 2>/dev/null; echo "0 * * * * $PROJECT_DIR/scripts/hourly_monitor.sh >> $LOG_DIR/hourly_monitor.log 2>&1") | crontab -
        log "Hourly monitoring cron added"
    fi
}

#--- Main startup sequence ---
main() {
    log "=========================================="
    log "TradingBingx Auto-Startup"
    log "=========================================="
    
    # Load environment
    cd "$PROJECT_DIR"
    source .env 2>/dev/null || true
    
    # Ensure live trading is armed
    export LIVE_TRACING_ARMED="yes"
    export TRADING_MODE="live"
    export HYPERLIQUID_MODE="live"
    export HL_SANDBOX="true"
    
    # Start services in order
    start_service "multi_streamer" "src/data/multi_streamer.py" "multi_streamer.py" "$LOG_DIR/streamer.log"
    sleep 5
    
    start_service "hl_executor" "src/execution/hl_executor.py" "hl_executor.py" "$LOG_DIR/polyarb.log"
    sleep 2
    
    start_service "hl_perp_ls" "-m src.execution.hl_perp_ls" "hl_perp_ls" "$LOG_DIR/perp_ls.log"
    sleep 2
    
    start_service "pairs_arb" "-m src.execution.pairs_arb_executor" "pairs_arb_executor" "$LOG_DIR/pairs_arb.log"
    sleep 2
    
    start_watchdog
    
    # Setup cron
    setup_cron
    
    log "=========================================="
    log "Startup complete!"
    log "=========================================="
    
    # Print status
    echo ""
    echo "Services started:"
    ps aux | grep -E "hl_executor|hl_perp_ls|pairs_arb|multi_streamer" | grep -v grep | awk '{print "  ✓ " $NF}'
    echo ""
    echo "Cron: $(crontab -l 2>/dev/null | grep -q 'hourly_monitor' && echo 'Active' || echo 'Disabled')"
    echo ""
}

# Run main
main
