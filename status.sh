#!/bin/bash
#==============================================================================
# TradingBingx Status Script
#==============================================================================
# Usage: ./status.sh
#==============================================================================

#--- Color codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#--- Project paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PID_DIR="$PROJECT_DIR/state/pids"
LOGS_DIR="$PROJECT_DIR/logs"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TradingBingx System Status              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

#--- Check services ---
echo -e "${YELLOW}Services:${NC}"

check_service() {
    local name=$1
    local pattern=$2
    
    if ps aux | grep "$pattern" | grep -v grep > /dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
    else
        echo -e "  ${RED}✗${NC} $name"
    fi
}

check_service "Multi-Streamer" "multi_streamer.py"
check_service "HL Executor" "hl_executor.py"
check_service "Perp L/S" "hl_perp_ls"
check_service "Pairs Arb" "pairs_arb_executor"
check_service "Watchdog" "watchdog.sh"

echo ""

#--- Check cron ---
echo -e "${YELLOW}Cron Monitoring:${NC}"
if crontab -l 2>/dev/null | grep -q "hourly_monitor.sh"; then
    echo -e "  ${GREEN}✓${NC} Hourly monitoring active"
else
    echo -e "  ${RED}✗${NC} Hourly monitoring disabled"
fi

echo ""

#--- Check equity ---
echo -e "${YELLOW}Equity:${NC}"
if [[ -f "$LOGS_DIR/polyarb.log" ]]; then
    equity=$(grep "Dynamic bankroll" "$LOGS_DIR/polyarb.log" | tail -1 | awk -F'\\$' '{print $2}')
    if [[ -n "$equity" ]]; then
        echo -e "  Current: ${GREEN}\$$equity${NC}"
    else
        echo -e "  ${YELLOW}No data available${NC}"
    fi
else
    echo -e "  ${YELLOW}Log file not found${NC}"
fi

echo ""

#--- Check positions ---
echo -e "${YELLOW}Positions:${NC}"
if [[ -f "$LOGS_DIR/polyarb.log" ]]; then
    last_trade=$(grep "LIVE ORDER SUCCESSFUL" "$LOGS_DIR/polyarb.log" | tail -1)
    if [[ -n "$last_trade" ]]; then
        echo -e "  Last trade: $(echo $last_trade | awk '{print $1, $2}')"
    else
        echo -e "  ${YELLOW}No trades executed${NC}"
    fi
fi

echo ""

#--- Recent logs ---
echo -e "${YELLOW}Recent Activity:${NC}"
if [[ -f "$LOGS_DIR/polyarb.log" ]]; then
    tail -3 "$LOGS_DIR/polyarb.log" | while read line; do
        echo "  $line"
    done
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""
