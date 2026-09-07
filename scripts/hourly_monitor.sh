#!/bin/bash
# Hourly System Monitor - TradingBingx
LOG_DIR="/home/prantik/Downloads/Personal/TradingBingx/logs"
REPORT_DIR="/home/prantik/Downloads/Personal/TradingBingx/reports"
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
grep "Dynamic bankroll" "$LOG_DIR/polyarb.log" | awk -F'updated to: \\$' '{print $2}' | awk 'BEGIN{min=99999;max=0;sum=0;n=0} {n++;sum+=$1; if($1>max)max=$1; if($1<min)min=$1} END{printf "Current: \$%.2f | Min: \$%.2f | Max: \$%.2f | Avg: \$%.2f | Drawdown: %.1f%%\n", $1, min, max, sum/n, ($1-max)/max*100}'
echo ""

# 3. Open Positions
echo "=== 3. OPEN POSITIONS ==="
grep "SETTLEMENT.*ROI" "$LOG_DIR/polyarb.log" | tail -3
echo ""

# 4. Trade Activity
echo "=== 4. TRADE ACTIVITY ==="
echo "Executions: $(grep 'EXECUTING VERIFIED' "$LOG_DIR/polyarb.log" | wc -l)"
echo "API rejections: $(grep 'API CALL ERROR' "$LOG_DIR/polyarb.log" | wc -l)"
echo "Risk rejections: $(grep 'REJECTED by Risk' "$LOG_DIR/polyarb.log" | wc -l)"
echo ""

# 5. Meta Learner
echo "=== 5. META LEARNER ==="
grep "META_LEARNER.*raw_roi" "$LOG_DIR/polyarb.log" | awk -F'raw_roi=' '{split($2,a,"%"); if(a[1]+0>0)pos++; else neg++} END{printf "Positive: %d | Negative: %d | Hit rate: %.1f%%\n", pos, neg, pos/(pos+neg)*100}'
echo ""

# 6. Bloodshed Check
echo "=== 6. BLOODSHED CHECK ==="
cat /home/prantik/Downloads/Personal/TradingBingx/data/calibration_ledger.jsonl | python3 -c "
import json, sys
trades = [json.loads(l) for l in sys.stdin if l.strip()]
if trades:
    rois = [t.get('outcome_roi',0) for t in trades]
    recent = rois[-10:] if len(rois)>=10 else rois
    print(f'All-time: Min={min(rois)*100:.2f}% Max={max(rois)*100:.2f}% Avg={sum(rois)/len(rois)*100:.2f}%')
    print(f'Recent 10: Min={min(recent)*100:.2f}% Max={max(recent)*100:.2f}% Avg={sum(recent)/len(recent)*100:.2f}%')
    if min(recent) < -0.05:
        print('!!! BLOODSHED WARNING: Recent loss > 5%')
else:
    print('No trade data')
" 2>/dev/null
echo ""

# 7. Last 5 Decisions
echo "=== 7. LAST 5 DECISIONS ==="
grep "Decision:" "$LOG_DIR/polyarb.log" | tail -5
echo ""

# 8. Perp L/S
echo "=== 8. PERP L/S ==="
grep "post-rebalance" "$LOG_DIR/perp_ls.log" | tail -2
echo ""

# 9. Pairs Arb
echo "=== 9. PAIRS ARB ==="
grep "check_pair" "$LOG_DIR/pairs_arb.log" | tail -1

} > "$REPORT" 2>&1

# Print report
cat "$REPORT"

# Keep only last 72 reports (3 days)
ls -t "$REPORT_DIR"/hourly_*.txt 2>/dev/null | tail -n +73 | xargs -r rm
