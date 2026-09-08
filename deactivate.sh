#!/bin/bash
#==============================================================================
# TradingBingx Deactivation Script
#==============================================================================
# Usage: ./deactivate.sh [OPTIONS]
#
# Options:
#   --keep-cron     Keep hourly monitoring cron active
#   --keep-streamer Keep multi-streamer running
#   --help          Show this help message
#==============================================================================

set -euo pipefail

#--- Color codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#--- Default values ---
KEEP_CRON=false
KEEP_STREAMER=false

#--- Project paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PID_DIR="$PROJECT_DIR/state/pids"
LOGS_DIR="$PROJECT_DIR/logs"

#--- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-cron)
            KEEP_CRON=true
            shift
            ;;
        --keep-streamer)
            KEEP_STREAMER=true
            shift
            ;;
        --help)
            head -15 "$0" | tail -10
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

#--- Helper functions ---
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

#--- Stop process by PID file ---
stop_process() {
    local name=$1
    local pid_file="$PID_DIR/$name.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_step "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            log_info "$name stopped"
        else
            log_warn "$name not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        log_warn "$name PID file not found"
    fi
}

#--- Main execution ---
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  TradingBingx Deactivation Script       ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
    echo ""
    
    # Stop trading processes
    log_step "Stopping trading processes..."
    stop_process "hl_executor"
    stop_process "perp_ls"
    stop_process "pairs_arb"
    stop_process "watchdog"
    
    # Stop streamer if not keeping
    if [[ "$KEEP_STREAMER" == false ]]; then
        stop_process "streamer"
    else
        log_info "Keeping multi-streamer running"
    fi
    
    # Remove cron if not keeping
    if [[ "$KEEP_CRON" == false ]]; then
        log_step "Removing hourly monitoring cron..."
        crontab -l 2>/dev/null | grep -v "hourly_monitor.sh" | crontab - 2>/dev/null || true
        log_info "Hourly monitoring cron removed"
    else
        log_info "Keeping hourly monitoring cron"
    fi
    
    # Clean up stale wake halt marker
    rm -f "$PROJECT_DIR/state/STALE_WAKE_HALT"
    
    echo ""
    log_info "System deactivated successfully!"
    echo ""
}

# Run main function
main
