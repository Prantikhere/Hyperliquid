# Data Flow Diagram

**Date:** Wed Sep 16, 2026
**Version:** 2.1 (Post-Fix)

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           HIGH-LEVEL DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   External      │     │   Internal      │     │   Storage       │
│   Sources       │     │   Processing    │     │   Systems       │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│                 │     │                 │     │                 │
│  Binance API ───┼────►│  Multi-Streamer ─┼────►│  Price DB       │
│                 │     │                 │     │  (SQLite)       │
│  HL API ────────┼────►│  HL Executor ────┼────►│  Trade DB       │
│                 │     │                 │     │  (SQLite)       │
│  WebSocket ─────┼────►│  Market Data ────┼────►│  State DB       │
│                 │     │                 │     │  (SQLite)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Intelligence   │
                        │  Layer          │
                        ├─────────────────┤
                        │                 │
                        │  Strategy ──────┼────► Quant Signals
                        │  Ensemble       │
                        │                 │
                        │  Regime ────────┼────► Market Regime
                        │  Engine         │
                        │                 │
                        │  Meta-Learner ──┼────► Confidence
                        │                 │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Decision       │
                        │  Layer          │
                        ├─────────────────┤
                        │                 │
                        │  Supervisor ────┼────► Trade Decision
                        │  Agent          │
                        │                 │
                        │  Risk ──────────┼────► Trade Size
                        │  Agent          │
                        │                 │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Execution      │
                        │  Layer          │
                        ├─────────────────┤
                        │                 │
                        │  Execution ─────┼────► Order
                        │  Agent          │
                        │                 │
                        │  HL Executor ───┼────► Position
                        │                 │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Settlement     │
                        │  Layer          │
                        ├─────────────────┤
                        │                 │
                        │  Settlement ────┼────► Position Closure
                        │  Agent          │
                        │                 │
                        │  Learning ──────┼────► Trade Analysis
                        │  Module         │
                        │                 │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Monitoring     │
                        │  Layer          │
                        ├─────────────────┤
                        │                 │
                        │  Hourly ────────┼────► System Status
                        │  Monitor        │
                        │                 │
                        │  Watchdog ──────┼────► Health Check
                        │  Process        │
                        │                 │
                        └─────────────────┘
```

---

## Detailed Data Flow by Component

### 1. Data Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Binance       │     │   Multi-        │     │   Price         │
│   WebSocket     │────►│   Streamer      │────►│   Database      │
│   (ETH, BTC,    │     │                 │     │                 │
│    SOL, etc.)   │     │  - Connects to  │     │  - Stores OHLCV │
│                 │     │    13 WebSocket  │     │  - Real-time    │
│                 │     │    streams      │     │    updates      │
│                 │     │  - Normalizes   │     │  - Historical   │
│                 │     │    data format  │     │    data         │
│                 │     │  - Handles      │     │                 │
│                 │     │    reconnection │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Order Book    │
                        │   Data          │
                        ├─────────────────┤
                        │                 │
                        │  - Bid/Ask      │
                        │    prices       │
                        │  - Depth        │
                        │  - Imbalance    │
                        │    calculation  │
                        │                 │
                        └─────────────────┘
```

### 2. Signal Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SIGNAL GENERATION FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Price         │     │   Strategy      │     │   Quant         │
│   Database      │────►│   Ensemble      │────►│   Signals       │
│                 │     │                 │     │                 │
│  - Historical   │     │  - Mean         │     │  - MR: 0.45     │
│    prices       │     │    Reversion    │     │  - MOM: 0.62    │
│  - Current      │     │    (Z-Score)    │     │  - OF: 0.51     │
│    price        │     │  - Momentum     │     │  - TREND: 0.58  │
│  - Volume       │     │    (RSI/MACD)   │     │  - RNN: 0.22    │
│                 │     │  - Order Flow   │     │                 │
│                 │     │    (Imbalance)  │     │                 │
│                 │     │  - Trend        │     │                 │
│                 │     │    (EMA)        │     │                 │
│                 │     │  - RNN          │     │                 │
│                 │     │    (LSTM)       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Composite     │
                        │   Score         │
                        ├─────────────────┤
                        │                 │
                        │  Formula:       │
                        │  (MR×0.25 +     │
                        │   MOM×0.25 +    │
                        │   OF×0.20 +     │
                        │   TREND×0.20 +  │
                        │   RNN×0.10)     │
                        │                 │
                        │  Output: 0-1    │
                        │                 │
                        └─────────────────┘
```

### 3. Regime Detection Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           REGIME DETECTION FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Price         │     │   Regime        │     │   Market        │
│   Database      │────►│   Engine        │────►│   Regime        │
│                 │     │                 │     │                 │
│  - 20-period    │     │  - Kaufman      │     │  - TRENDING     │
│    window       │     │    Efficiency   │     │  - MEAN_REVERTING│
│  - Returns      │     │    Ratio (ER)   │     │  - NEUTRAL      │
│  - Volatility   │     │  - Volatility   │     │  - HIGH_VOLATILITY│
│                 │     │    Ratio        │     │                 │
│                 │     │  - Trend        │     │                 │
│                 │     │    Detection    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Regime        │
                        │   Classification│
                        ├─────────────────┤
                        │                 │
                        │  ER > 0.30:     │
                        │  TRENDING       │
                        │                 │
                        │  ER < 0.15:     │
                        │  MEAN_REVERTING │
                        │                 │
                        │  0.15 ≤ ER ≤ 0.30:│
                        │  NEUTRAL        │
                        │                 │
                        │  VolRatio > 1.5:│
                        │  HIGH_VOLATILITY│
                        │                 │
                        └─────────────────┘
```

### 4. Decision Making Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DECISION MAKING FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Composite     │     │   Supervisor    │     │   Trade         │
│   Score         │────►│   Agent         │────►│   Decision      │
│                 │     │                 │     │                 │
│  - 0.70-1.00:   │     │  - Check        │     │  - BUY          │
│    Strong BUY   │     │    whitelist    │     │  - SELL         │
│  - 0.55-0.70:   │     │  - Check        │     │  - HOLD         │
│    Weak BUY     │     │    regime       │     │                 │
│  - 0.45-0.55:   │     │  - Apply        │     │                 │
│    Neutral      │     │    entry logic  │     │                 │
│  - 0.30-0.45:   │     │  - Override     │     │                 │
│    Weak SELL    │     │    for strong   │     │                 │
│  - 0.00-0.30:   │     │    signals      │     │                 │
│    Strong SELL  │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Entry Logic   │
                        │   (Updated)     │
                        ├─────────────────┤
                        │                 │
                        │  BUY if:        │
                        │  - composite    │
                        │    > 0.55 AND   │
                        │  - uptrend AND  │
                        │  - symbol_ok    │
                        │                 │
                        │  OR if:         │
                        │  - composite    │
                        │    >= 0.70 AND  │
                        │  - (MEAN_REV OR │
                        │    NEUTRAL) AND │
                        │  - symbol_ok    │
                        │                 │
                        │  SELL if:       │
                        │  - composite    │
                        │    < 0.45 AND   │
                        │  - downtrend AND│
                        │  - symbol_ok    │
                        │                 │
                        │  OR if:         │
                        │  - composite    │
                        │    <= 0.30 AND  │
                        │  - (MEAN_REV OR │
                        │    NEUTRAL) AND │
                        │  - symbol_ok    │
                        │                 │
                        └─────────────────┘
```

### 5. Risk Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RISK MANAGEMENT FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Trade         │     │   Risk          │     │   Risk-Adjusted │
│   Decision      │────►│   Agent         │────►│   Trade Size    │
│                 │     │                 │     │                 │
│  - BUY/SELL     │     │  - Kelly        │     │  - Position     │
│  - Confidence   │     │    Criterion    │     │    Sizing       │
│  - Symbol       │     │  - Margin Check │     │  - USD Amount   │
│                 │     │  - Min Order    │     │  - Leverage     │
│                 │     │  - Position     │     │                 │
│                 │     │    Limits       │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Kelly         │
                        │   Criterion     │
                        ├─────────────────┤
                        │                 │
                        │  Formula:       │
                        │  f* = (p × b - q) / b│
                        │                 │
                        │  Where:         │
                        │  p = win rate   │
                        │  b = win/loss   │
                        │      ratio      │
                        │  q = 1 - p      │
                        │                 │
                        │  Kelly fraction │
                        │  = 0.20         │
                        │                 │
                        └─────────────────┘
```

### 6. Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Risk-Adjusted │     │   Execution     │     │   HL Executor   │
│   Trade Size    │────►│   Agent         │────►│                 │
│                 │     │                 │     │  - Connect to   │
│  - Symbol       │     │  - Build order  │     │    HL testnet   │
│  - Direction    │     │  - Route to     │     │  - Submit order │
│  - Size (USD)   │     │    exchange     │     │  - Handle       │
│  - Confidence   │     │  - Handle       │     │    response     │
│                 │     │    latency      │     │  - Update state │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Order Types   │
                        ├─────────────────┤
                        │                 │
                        │  - Market Order │
                        │    (Immediate)  │
                        │                 │
                        │  - Limit Order  │
                        │    (Price       │
                        │     specific)   │
                        │                 │
                        └─────────────────┘
```

### 7. Settlement Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SETTLEMENT FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Position      │     │   Settlement    │     │   Position      │
│   Monitoring    │────►│   Agent         │────►│   Closure       │
│                 │     │                 │     │                 │
│  - Unrealized   │     │  - Hard Stop    │     │  - Market Order │
│    PnL          │     │    (0.6-2.0%)   │     │  - Partial      │
│  - Duration     │     │  - Dynamic Stop │     │    Closure      │
│  - ROI          │     │  - Breakeven    │     │  - Full         │
│                 │     │  - Trailing     │     │    Closure      │
│                 │     │  - Profit Lock  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Exit Triggers │
                        ├─────────────────┤
                        │                 │
                        │  1. Hard Stop   │
                        │     Loss        │
                        │     (0.6-2.0%)  │
                        │                 │
                        │  2. Dynamic     │
                        │     Stop        │
                        │     (Tightens   │
                        │      at 1%→1.5%,│
                        │      2%→2%)     │
                        │                 │
                        │  3. Breakeven   │
                        │     Stop        │
                        │     (ROI > 2%)  │
                        │                 │
                        │  4. Trailing    │
                        │     Stop        │
                        │     (15% from   │
                        │      peak)      │
                        │                 │
                        │  5. High Profit │
                        │     Lock        │
                        │     (10% from   │
                        │      peak at    │
                        │      >4% ROI)   │
                        │                 │
                        │  6. Regime      │
                        │     Exit        │
                        │     (LONG in    │
                        │      HIGH_VOL)  │
                        │                 │
                        │  7. Momentum    │
                        │     Exit        │
                        │     (ROI > 2%   │
                        │      comp < 0.35│
                        │      fade)      │
                        │                 │
                        │  8. Stale       │
                        │     Position    │
                        │     Exit        │
                        │     (4h, < 5%   │
                        │      ROI)       │
                        │                 │
                        └─────────────────┘
```

### 8. Learning Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           LEARNING FLOW                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Trade         │     │   Learning      │     │   Trade         │
│   Execution     │────►│   Module        │────►│   Analysis      │
│                 │     │                 │     │                 │
│  - Entry price  │     │  - Record       │     │  - Win/Loss     │
│  - Exit price   │     │    outcome      │     │  - ROI          │
│  - Direction    │     │  - Track        │     │  - Duration     │
│  - Size         │     │    failures     │     │  - Regime       │
│                 │     │  - Identify     │     │  - Confidence   │
│                 │     │    patterns     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Failure       │
                        │   Patterns      │
                        ├─────────────────┤
                        │                 │
                        │  - Large Loss   │
                        │    (>2% loss)   │
                        │                 │
                        │  - Wrong Regime │
                        │    (Trend trade │
                        │     in sideways)│
                        │                 │
                        │  - Bad Timing   │
                        │    (Entry at    │
                        │     peak)       │
                        │                 │
                        │  - High         │
                        │    Confidence   │
                        │    Loss         │
                        │    (Conf > 0.6  │
                        │     but loss)   │
                        │                 │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Parameter     │
                        │   Adjustment    │
                        ├─────────────────┤
                        │                 │
                        │  - Tighten SL   │
                        │    for repeat   │
                        │    failures     │
                        │                 │
                        │  - Reduce       │
                        │    confidence   │
                        │    for bad      │
                        │    patterns     │
                        │                 │
                        │  - Avoid        │
                        │    similar      │
                        │    trades       │
                        │                 │
                        └─────────────────┘
```

### 9. Monitoring Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MONITORING FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   System        │     │   Hourly        │     │   System        │
│   Status        │────►│   Monitor       │────►│   Report        │
│                 │     │                 │     │                 │
│  - Capital      │     │  - Capital      │     │  - Growth       │
│  - Positions    │     │    tracking     │     │    rate         │
│  - Trades       │     │  - Position     │     │  - Win rate     │
│  - Performance  │     │    monitor      │     │  - Expectancy   │
│                 │     │  - Trade        │     │  - Alerts       │
│                 │     │    analysis     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Watchdog      │
                        │   Process       │
                        ├─────────────────┤
                        │                 │
                        │  - Every 2 min  │
                        │  - Check PID    │
                        │  - Auto-restart │
                        │  - Health check │
                        │                 │
                        └─────────────────┘
```

### 10. Storage Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE FLOW                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Application   │     │   Database      │     │   File System   │
│   Layer         │────►│   Layer         │────►│   Layer         │
│                 │     │                 │     │                 │
│  - Trade data   │     │  - SQLite       │     │  - JSONL files  │
│  - Price data   │     │  - Prices       │     │  - Calibration  │
│  - State data   │     │  - Trades       │     │  - Learning     │
│                 │     │  - States       │     │  - Logs         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   Data Types    │
                        ├─────────────────┤
                        │                 │
                        │  SQLite:        │
                        │  - prices.db    │
                        │  - trades.db    │
                        │  - states.db    │
                        │                 │
                        │  JSONL:         │
                        │  - calibration  │
                        │    _ledger.jsonl│
                        │  - learning     │
                        │    _outcomes.   │
                        │    jsonl        │
                        │                 │
                        │  Models:        │
                        │  - rnn_price    │
                        │    _predictor.  │
                        │    pth          │
                        │  - meta_learner │
                        │    .json        │
                        │                 │
                        └─────────────────┘
```

---

## Data Formats

### Price Data

```json
{
  "symbol": "ETH/USDT",
  "timestamp": "2026-09-16T19:30:00Z",
  "open": 3200.50,
  "high": 3210.00,
  "low": 3195.00,
  "close": 3205.00,
  "volume": 1250.50
}
```

### Quant Signals

```json
{
  "symbol": "SUI/USDT",
  "mean_reversion": 0.45,
  "momentum": 0.62,
  "order_flow": 0.51,
  "trend": 0.58,
  "rnn": 0.22,
  "composite_score": 0.48,
  "regime": "MEAN_REVERTING"
}
```

### Trade Decision

```json
{
  "symbol": "SUI/USDT",
  "action": "BUY",
  "confidence": 0.48,
  "composite_score": 0.70,
  "regime": "MEAN_REVERTING",
  "risk_adjusted_size": 5.20,
  "entry_price": 1.85
}
```

### Position State

```json
{
  "symbol": "ETC/USDT",
  "direction": "LONG",
  "entry_price": 18.50,
  "current_price": 19.20,
  "quantity": 21.1,
  "unrealized_pnl": 14.77,
  "roi": 3.78,
  "duration_hours": 2.5,
  "stop_loss": 18.13,
  "take_profit": 22.20
}
```

---

## Data Flow Metrics

| Flow | Frequency | Volume | Latency |
|------|-----------|--------|---------|
| **Price Ingestion** | Real-time | 1000+ ticks/sec | < 10ms |
| **Signal Generation** | Every 60s | 20 symbols | < 100ms |
| **Regime Detection** | Every 60s | 20 symbols | < 50ms |
| **Decision Making** | Every 60s | 20 symbols | < 200ms |
| **Risk Evaluation** | Every 60s | Per trade | < 50ms |
| **Order Execution** | On demand | Per trade | < 500ms |
| **Position Monitoring** | Real-time | 2-5 positions | < 100ms |
| **Settlement Check** | Every 10s | 2-5 positions | < 100ms |
| **Learning Recording** | On trade close | Per trade | < 10ms |
| **System Monitoring** | Every hour | System-wide | < 5s |

---

**Data Flow Version:** 2.1
**Last Updated:** Wed Sep 16, 2026
**Status:** Operational
