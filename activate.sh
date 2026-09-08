#!/bin/bash
#==============================================================================
# TradingBingx Master Activation Script
#==============================================================================
# Usage: ./activate.sh [OPTIONS]
#
# Options:
#   --env ENV           Environment: testnet (default), live, paper
#   --exchange EXCHANGE Exchange: hyperliquid (default), bingx, both
#   --mode MODE         Mode: full (default), executor, streamer, monitor
#   --bankroll AMOUNT   Starting bankroll in USD (default: from .env)
#   --log-level LEVEL   Log level: INFO (default), DEBUG, WARNING
#   --no-cron           Disable hourly monitoring cron
#   --force             Force restart (kill existing processes)
#   --help              Show this help message
#
# Examples:
#   ./activate.sh                          # Start full system in testnet
#   ./activate.sh --env paper --exchange hyperliquid
#   ./activate.sh --mode executor --force  # Force restart executor only
#   ./activate.sh --bankroll 500 --log-level DEBUG
#==============================================================================

set -euo pipefail

#--- Color codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#--- Default values ---
ENV="testnet"
EXCHANGE="hyperliquid"
MODE="full"
BANKROLL=""
LOG_LEVEL="INFO"
NO_CRON=false
FORCE=false

#--- Project paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"
LOGS_DIR="$PROJECT_DIR/logs"
STATE_DIR="$PROJECT_DIR/state"
PID_DIR="$PROJECT_DIR/state/pids"

#--- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --exchange)
            EXCHANGE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --bankroll)
            BANKROLL="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --no-cron)
            NO_CRON=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -20
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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

#--- Pre-flight checks ---
preflight_checks() {
    log_step "Running pre-flight checks..."
    
    # Check Python venv
    if [[ ! -f "$VENV_PYTHON" ]]; then
        log_error "Python virtual environment not found at $VENV_PYTHON"
        log_info "Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
        exit 1
    fi
    
    # Check .env file
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
        log_error ".env file not found"
        log_info "Create .env with required API keys"
        exit 1
    fi
    
    # Check required directories
    mkdir -p "$LOGS_DIR" "$STATE_DIR" "$PID_DIR"
    
    # Check required env vars
    source "$PROJECT_DIR/.env"
    
    if [[ "$EXCHANGE" == "hyperliquid" ]] || [[ "$EXCHANGE" == "both" ]]; then
        if [[ -z "${HL_PRIVATE_KEY:-}" ]] && [[ -z "${HL_PRIVATE:-}" ]]; then
            log_error "HL_PRIVATE_KEY or HL_PRIVATE not set in .env"
            exit 1
        fi
        if [[ -z "${HL_WALLET_ADDRESS:-}" ]]; then
            log_error "HL_WALLET_ADDRESS not set in .env"
            exit 1
        fi
    fi
    
    if [[ "$EXCHANGE" == "bingx" ]] || [[ "$EXCHANGE" == "both" ]]; then
        if [[ -z "${BINGX_API_KEY:-}" ]]; then
            log_error "BINGX_API_KEY not set in .env"
            exit 1
        fi
    fi
    
    log_info "Pre-flight checks passed"
}

#--- Kill existing processes ---
kill_existing() {
    if [[ "$FORCE" == true ]]; then
        log_warn "Force mode: Killing existing processes..."
        
        # Kill trading processes
        pkill -f "hl_executor.py" 2>/dev/null || true
        pkill -f "hl_perp_ls" 2>/dev/null || true
        pkill -f "pairs_arb_executor" 2>/dev/null || true
        pkill -f "multi_streamer.py" 2>/dev/null || true
        pkill -f "watchdog.sh" 2>/dev/null || true
        
        sleep 2
        log_info "Existing processes killed"
    else
        # Check if already running
        if ps aux | grep -E "hl_executor|hl_perp_ls|pairs_arb" | grep -v grep > /dev/null; then
            log_warn "Trading processes already running. Use --force to restart."
            exit 1
        fi
    fi
}

#--- Start multi-streamer ---
start_streamer() {
    log_step "Starting multi-streamer..."
    
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -u src/data/multi_streamer.py >> "$LOGS_DIR/streamer.log" 2>&1 &
    echo $! > "$PID_DIR/streamer.pid"
    
    log_info "Multi-streamer started (PID: $(cat $PID_DIR/streamer.pid))"
}

#--- Start HL Executor ---
start_hl_executor() {
    log_step "Starting HL Executor..."
    
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -u src/execution/hl_executor.py >> "$LOGS_DIR/polyarb.log" 2>&1 &
    echo $! > "$PID_DIR/hl_executor.pid"
    
    log_info "HL Executor started (PID: $(cat $PID_DIR/hl_executor.pid))"
}

#--- Start Perp L/S ---
start_perp_ls() {
    log_step "Starting Perp L/S..."
    
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -u -m src.execution.hl_perp_ls >> "$LOGS_DIR/perp_ls.log" 2>&1 &
    echo $! > "$PID_DIR/perp_ls.pid"
    
    log_info "Perp L/S started (PID: $(cat $PID_DIR/perp_ls.pid))"
}

#--- Start Pairs Arb ---
start_pairs_arb() {
    log_step "Starting Pairs Arb Executor..."
    
    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -u -m src.execution.pairs_arb_executor >> "$LOGS_DIR/pairs_arb.log" 2>&1 &
    echo $! > "$PID_DIR/pairs_arb.pid"
    
    log_info "Pairs Arb Executor started (PID: $(cat $PID_DIR/pairs_arb.pid))"
}

#--- Start Watchdog ---
start_watchdog() {
    log_step "Starting Watchdog..."
    
    cd "$PROJECT_DIR"
    nohup bash watchdog.sh >> /dev/null 2>&1 &
    echo $! > "$PID_DIR/watchdog.pid"
    
    log_info "Watchdog started (PID: $(cat $PID_DIR/watchdog.pid))"
}

#--- Setup cron monitoring ---
setup_cron() {
    if [[ "$NO_CRON" == true ]]; then
        log_warn "Cron monitoring disabled (--no-cron flag)"
        return
    fi
    
    log_step "Setting up hourly monitoring cron..."
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "hourly_monitor.sh"; then
        log_info "Hourly monitoring cron already active"
    else
        # Add cron job
        (crontab -l 2>/dev/null; echo "0 * * * * $PROJECT_DIR/scripts/hourly_monitor.sh >> $LOGS_DIR/hourly_monitor.log 2>&1") | crontab -
        log_info "Hourly monitoring cron added"
    fi
}

#--- Set environment variables ---
set_env_vars() {
    log_step "Setting environment variables..."
    
    # Set trading mode based on environment
    case "$ENV" in
        testnet)
            export TRADING_MODE="live"
            export HYPERLIQUID_MODE="live"
            export HL_SANDBOX="true"
            ;;
        live)
            export TRADING_MODE="live"
            export HYPERLIQUID_MODE="live"
            export HL_SANDBOX="false"
            ;;
        paper)
            export TRADING_MODE="paper"
            export HYPERLIQUID_MODE="paper"
            ;;
    esac
    
    # Set log level
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Set bankroll if provided
    if [[ -n "$BANKROLL" ]]; then
        export BANKROLL="$BANKROLL"
    fi
    
    # Ensure live trading is armed
    export LIVE_TRADING_ARMED="yes"
    
    log_info "Environment: $ENV | Exchange: $EXCHANGE | Mode: $MODE"
}

#--- Print status ---
print_status() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  TradingBingx System Activated${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "  Environment: ${BLUE}$ENV${NC}"
    echo -e "  Exchange:    ${BLUE}$EXCHANGE${NC}"
    echo -e "  Mode:        ${BLUE}$MODE${NC}"
    echo -e "  Log Level:   ${BLUE}$LOG_LEVEL${NC}"
    echo ""
    echo -e "  ${YELLOW}Services:${NC}"
    ps aux | grep -E "hl_executor|hl_perp_ls|pairs_arb|multi_streamer" | grep -v grep | awk '{print "    - " $NF}' || echo "    - Starting..."
    echo ""
    echo -e "  ${YELLOW}Cron Monitoring:${NC}"
    if crontab -l 2>/dev/null | grep -q "hourly_monitor.sh"; then
        echo "    - Active (every hour)"
    else
        echo "    - Disabled"
    fi
    echo ""
    echo -e "  ${YELLOW}Logs:${NC}"
    echo "    - $LOGS_DIR/polyarb.log"
    echo "    - $LOGS_DIR/perp_ls.log"
    echo "    - $LOGS_DIR/pairs_arb.log"
    echo "    - $LOGS_DIR/streamer.log"
    echo "    - $LOGS_DIR/hourly_monitor.log"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

#--- Main execution ---
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  TradingBingx Activation Script         ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
    echo ""
    
    # Run pre-flight checks
    preflight_checks
    
    # Kill existing processes if force mode
    kill_existing
    
    # Set environment variables
    set_env_vars
    
    # Start services based on mode
    case "$MODE" in
        full)
            start_streamer
            sleep 2
            start_hl_executor
            start_perp_ls
            start_pairs_arb
            start_watchdog
            ;;
        executor)
            start_hl_executor
            start_perp_ls
            start_pairs_arb
            start_watchdog
            ;;
        streamer)
            start_streamer
            ;;
        monitor)
            start_watchdog
            ;;
        *)
            log_error "Unknown mode: $MODE"
            exit 1
            ;;
    esac
    
    # Setup cron monitoring
    setup_cron
    
    # Print status
    print_status
    
    log_info "System activated successfully!"
    log_info "Monitor with: tail -f $LOGS_DIR/polyarb.log"
}

# Run main function
main
