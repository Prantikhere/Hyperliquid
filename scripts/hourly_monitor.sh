#!/bin/bash
# Hourly System Monitor - TradingBingx (Simplified)
LOG_DIR="/home/prantik/Downloads/Personal/TradingBingx/logs"
REPORT_DIR="/home/prantik/Downloads/Personal/TradingBingx/reports"
HL_WALLET="0x07dd93729632BEF8B3A522F8079efD483990aE34"
mkdir -p "$REPORT_DIR"

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M')
REPORT="$REPORT_DIR/hourly_$TIMESTAMP.txt"

{
echo "=========================================="
echo " HOURLY ASSESSMENT REPORT"
echo " $(date)"
echo "=========================================="
echo ""

# 1. Services
echo "=== 1. SERVICES ==="
ps aux | grep -E "hl_executor|hl_perp|pairs_arb|streamer" | grep -v grep | awk '{print "  PID="$2, "START="$9, "CMD="$NF}'
echo ""

# 2. Equity / AUM
echo "=== 2. EQUITY / AUM ==="
curl -s --max-time 10 -X POST https://api.hyperliquid-testnet.xyz/info \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"clearinghouseState\",\"user\":\"$HL_WALLET\"}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
margin = data.get('marginSummary', {})
account_value = float(margin.get('accountValue', 0))
margin_used = float(margin.get('totalMarginUsed', 0))
free_margin = account_value - margin_used
margin_pct = (margin_used / account_value * 100) if account_value > 0 else 0

print(f'  Account Value:    \${account_value:.2f}')
print(f'  Margin Used:      \${margin_used:.2f} ({margin_pct:.1f}%)')
print(f'  Free Margin:      \${free_margin:.2f}')

start_equity = 195.16
growth = account_value - start_equity
growth_pct = (growth / start_equity * 100) if start_equity > 0 else 0
print(f'  Starting Equity:  \$195.16')
print(f'  Growth:           +\${growth:.2f} (+{growth_pct:.2f}%)')
" 2>/dev/null
echo ""

# 3. Open Positions
echo "=== 3. OPEN POSITIONS ==="
curl -s --max-time 10 -X POST https://api.hyperliquid-testnet.xyz/info \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"clearinghouseState\",\"user\":\"$HL_WALLET\"}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
positions = data.get('assetPositions', [])
total_pnl = 0
for p in positions:
    pos = p.get('position', {})
    coin = pos.get('coin', '')
    szi = float(pos.get('szi', 0))
    pnl = float(pos.get('unrealizedPnl', 0))
    if szi != 0:
        total_pnl += pnl
        direction = 'LONG' if szi > 0 else 'SHORT'
        print(f'  {coin}: {abs(szi):.1f} {direction}, PnL=\${pnl:+.2f}')
print(f'  Total PnL: \${total_pnl:+.2f}')
" 2>/dev/null
echo ""

# 4. Trade Activity
echo "=== 4. TRADE ACTIVITY ==="
today=$(date '+%Y-%m-%d')
echo "  Executions: $(grep 'LIVE ORDER SUCCESSFUL' "$LOG_DIR/polyarb.log" 2>/dev/null | grep "$today" | wc -l)"
echo "  Risk Rejections: $(grep 'REJECTED by Risk' "$LOG_DIR/polyarb.log" 2>/dev/null | grep "$today" | wc -l)"
echo ""

# 5. Performance
echo "=== 5. PERFORMANCE ==="
grep "Dynamic bankroll" "$LOG_DIR/polyarb.log" 2>/dev/null | tail -1 | awk -F'updated to: \\$' '{printf "  Current: \$%s\n", $2}'
echo ""

# 6. Last Decision
echo "=== 6. LAST DECISION ==="
grep "Decision:" "$LOG_DIR/polyarb.log" 2>/dev/null | tail -1
echo ""

echo "=========================================="
echo " Report: $REPORT"
echo "=========================================="

} > "$REPORT" 2>&1

cat "$REPORT"

# Keep only last 72 reports (3 days)
ls -t "$REPORT_DIR"/hourly_*.txt 2>/dev/null | tail -n +73 | xargs -r rm 2>/dev/null
