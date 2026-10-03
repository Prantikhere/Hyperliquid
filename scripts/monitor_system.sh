#!/usr/bin/env bash
# System health monitor — run periodically to catch significant incidents
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==============================="
echo "SYSTEM HEALTH REPORT — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "==============================="

# 1. Process health
echo ""
echo "--- PROCESS STATUS ---"
NEED_HEAL=0
if pgrep -a -f "hl_executor" >/dev/null 2>&1; then
    echo "✅ hl_executor: RUNNING"
else
    echo "❌ hl_executor: DOWN"
    NEED_HEAL=1
fi
if pgrep -a -f "hl_perp_ls" >/dev/null 2>&1; then
    echo "✅ perp_ls: RUNNING"
else
    echo "⚠️  perp_ls: DOWN"
    NEED_HEAL=1
fi
if pgrep -a -f "pairs_arb" >/dev/null 2>&1; then
    echo "✅ pairs_arb: RUNNING"
else
    echo "⏸️  pairs_arb: PAUSED (100% Margin Dedicated to Scalping)"
fi

if pgrep -a -f "watchdog" >/dev/null 2>&1; then
    echo "✅ watchdog: RUNNING"
else
    echo "⚠️  watchdog: DOWN"
    NEED_HEAL=1
fi

if [ "$NEED_HEAL" -eq 1 ]; then
    echo "⚡ Auto-healing triggered: invoking watchdog.sh to restore down processes..."
    /bin/bash ./watchdog.sh >/dev/null 2>&1 || true
fi

# 2. Laya health
echo ""
echo "--- LAYA SUPERVISION ---"
LAYA=$(curl -s --max-time 5 http://127.0.0.1:8080/health 2>/dev/null || echo "DOWN")
if echo "$LAYA" | grep -q '"status":"ok"'; then
    FEEDBACK=$(echo "$LAYA" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'v{d.get(\"laya_version\",\"?\")}, feedback={d.get(\"total_feedback_records\",0)}, decisions={d.get(\"total_historical_decisions\",0)}')" 2>/dev/null || echo "parse error")
    echo "✅ Laya: UP — $FEEDBACK"
else
    echo "❌ Laya: DOWN — $LAYA"
fi

# 3. Dynamic whitelist
echo ""
echo "--- DYNAMIC WHITELIST ---"
./venv/bin/python3 -c "
import redis, json
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
cached = r.get('dynamic_whitelist:symbols:hyperliquid')
if cached:
    wl = json.loads(cached)
    toxic = {'NEAR/USDT','FIL/USDT','HBAR/USDT'}
    leaked = [s for s in wl if s in toxic]
    print(f'Active: {len(wl)} symbols')
    if leaked:
        print(f'❌ TOXIC LEAK: {leaked}')
    else:
        print('✅ No toxic symbols leaked')
    print(f'Symbols: {wl}')
else:
    print('⚠️  No cached whitelist (will compute on next cycle)')
" 2>/dev/null

# 4. On-chain positions & equity
echo ""
echo "--- ACCOUNT STATUS ---"
./venv/bin/python3 -c "
from src.execution.hl_raw import HlSdkClient
client = HlSdkClient()
state = client.info.user_state(client.wallet_address)
margin = state.get('marginSummary', {})
equity = float(margin.get('accountValue', 0))
margin_used = float(margin.get('totalMarginUsed', 0))
positions = [p for p in state.get('assetPositions', []) if float(p['position']['szi']) != 0]
print(f'Equity: \${equity:.2f} | Margin used: \${margin_used:.2f} | Open positions: {len(positions)}')
for pos in positions:
    p = pos['position']
    print(f'  {p[\"coin\"]}: size={p[\"szi\"]}, entry={p[\"entryPx\"]}, uPnl={p[\"unrealizedPnl\"]}')
" 2>/dev/null

# 5. Recent errors (last 30 min)
echo ""
echo "--- RECENT ERRORS (30 min) ---"
LATEST_LOGS=$(ls -t logs/polyarb*.log 2>/dev/null | head -3)
LATEST_LOGS=${LATEST_LOGS:-logs/polyarb.log}
RECENT_ERRORS=$(awk -v cutoff="$(date -d '30 min ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /ERROR|CRASH|Fatal/ {c++} END {print c+0}' $LATEST_LOGS 2>/dev/null || echo 0)
echo "Errors in last 30 min: $RECENT_ERRORS"
if [ "$RECENT_ERRORS" -gt 0 ]; then
    awk -v cutoff="$(date -d '30 min ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /ERROR|CRASH|Fatal/' $LATEST_LOGS 2>/dev/null | tail -5
fi

# 6. Recent Laya vetoes (last 1h)
echo ""
echo "--- LAYA DECISIONS (last 1h) ---"
SUPERVISIONS=$(awk -v cutoff="$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /LAYA_SUPERVISION.*Candidate/ {c++} END {print c+0}' $LATEST_LOGS 2>/dev/null || echo 0)
VETOES=$(awk -v cutoff="$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /LAYA_VETO|LAYA_SANITY/ {c++} END {print c+0}' $LATEST_LOGS 2>/dev/null || echo 0)
echo "Total supervisions: $SUPERVISIONS, vetoes: $VETOES"

# 7. Trade executions (last 1h)
echo ""
echo "--- TRADE ACTIVITY (last 1h) ---"
EXECUTIONS=$(awk -v cutoff="$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /EXECUTING VERIFIED TRADE/ {c++} END {print c+0}' $LATEST_LOGS 2>/dev/null || echo 0)
echo "Executed trades: $EXECUTIONS"
if [ "$EXECUTIONS" -gt 0 ]; then
    awk -v cutoff="$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')" '/^[0-9]{4}-[0-9]{2}-[0-9]{2}/ && $0 >= cutoff && /EXECUTING VERIFIED TRADE/' $LATEST_LOGS 2>/dev/null | head -10
fi

# 8. Universe feed health
echo ""
echo "--- UNIVERSE FEED ---"
ACTIVE_POLYARB=$(ls -t logs/polyarb*.log 2>/dev/null | head -1)
FEED_LINES=$(awk '/UNIVERSE_FEED.*Updated/ {c++} END {print c+0}' "$ACTIVE_POLYARB" 2>/dev/null || echo 0)
echo "Total feed cycles: $FEED_LINES"
# 9. Daily Calibration & Fine-Tuning Check
echo ""
echo "--- DAILY CALIBRATION & FINE-TUNING ---"
./venv/bin/python3 -c "
import os, time, json, redis
from datetime import datetime, timezone

r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
last_calib = r.get('system:last_daily_calibration_date')

if last_calib != today:
    print(f'⚡ Triggering daily calibration run for {today}...')
    import subprocess
    res = subprocess.run(['./venv/bin/python3', 'scripts/daily_calibration.py'], capture_output=True, text=True)
    r.set('system:last_daily_calibration_date', today)
    print(res.stdout)
    if res.stderr:
        print('Warnings/Errors:', res.stderr[:300])
else:
    print(f'✅ Calibration for today ({today}) already completed.')
    if os.path.exists('logs/calibration.log'):
        with open('logs/calibration.log', 'r') as f:
            json_lines = [l.strip() for l in f if l.strip().startswith('{')]
            if json_lines:
                try:
                    last_rep = json.loads(json_lines[-1])
                    print(f'  Last adjustments: {last_rep.get(\"adjustments\", {})}')
                    print(f'  Actions taken: {last_rep.get(\"actions_taken\", [])}')
                except Exception:
                    pass
" 2>&1 || true

echo ""
echo "==============================="
echo "END REPORT"
echo "==============================="

# Push desktop notification if notify-send is available
if command -v notify-send >/dev/null 2>&1; then
    EQ_STR=$(./venv/bin/python3 -c "
from src.execution.hl_raw import HlSdkClient
client = HlSdkClient()
s = client.info.user_state(client.wallet_address)
m = s.get('marginSummary', {})
e = float(m.get('accountValue', 0))
u = float(m.get('totalMarginUsed', 0))
print(f'Equity: \${e:.2f} | Margin: \${u:.2f}')
" 2>/dev/null || echo "TradingBingx Active")
    notify-send -u low "TradingBingx Surveillance" "$EQ_STR | All daemons active" 2>/dev/null || true
fi

