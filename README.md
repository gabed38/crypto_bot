# Crypto Trading Bot

An LLM-powered crypto trading bot that uses Claude to analyze markets and recommend trades, with fully mechanical position management. Built as a paper-trading POC — all trades are written to file, nothing is executed on an exchange.

**Core strategy:** Volatile swing trading — buy dips of 5–30% on mid/small-cap coins and ride the mean-reversion bounce, targeting 5–10% gains in hours to days.

## Quick Start

```bash
cd crypto_bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Build coin behavioral profiles (run once, then weekly)
python scripts/build_coin_profiles.py

# Run the bot
python scripts/intraday_trader.py

# Launch the dashboard
streamlit run dashboard/app.py
```

## How It Works

The bot runs a single script (`scripts/intraday_trader.py`) that executes three phases in sequence. It is designed to run every **4 hours** via cron.

```
┌─────────────────────────────────────────────────────────────────┐
│                    intraday_trader.py                            │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   PHASE 1    │──▶│   PHASE 2    │──▶│      PHASE 3       │  │
│  │  Manage      │   │  Discover    │   │  Analyze & Learn   │  │
│  │  Positions   │   │  New Trades  │   │  from Closed Trades│  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│   (mechanical,             ▲                      │             │
│    no LLM)          lessons.json ◀────────────────┘             │
│         ▼                                                       │
│  open_positions.json                                            │
│  resolved_trades.jsonl                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Position Management (Fully Mechanical)

Checks every open position using five deterministic price-based rules — **no LLM involved**. Rules are checked in priority order; the first rule that fires closes the position.

1. **Load open positions** from `data/positions/open_positions.json`
2. **Batch fetch current prices** from CoinGecko for all held coins
3. **Update trailing high-water marks** — track the highest observed price per position
4. **Apply exit rules in order:**

   | Rule | Trigger | Close Type |
   |------|---------|------------|
   | Stop-loss | Down ≥ `adaptive_stop_pct` from entry | `STOP_LOSS` |
   | Trailing stop | Down ≥ `adaptive_stop_pct` from high-water mark (once in profit) | `TRAILING_STOP` |
   | Take-profit | Gain ≥ `target_pct` | `TAKE_PROFIT` |
   | Profit protection | Peak gain ≥ 4%, current gain ≤ 50% of peak (e.g. peak +8% → close at +4%) | `PROFIT_PROTECTION` |
   | Time horizon expired | Held past `time_horizon` | `TIME_EXPIRED` |

5. **Execute closes** — closed positions are appended to `data/positions/resolved_trades.jsonl`
6. **Update open positions** — surviving positions saved back with updated prices and P&L

#### Vol-adaptive stops

Stop-loss percentages are not flat — they're computed per coin from 14-day daily volatility:

```
adaptive_stop_pct = max(min(daily_vol_pct × 3, 25.0), 10.0)
```

| Coin type | Daily vol | Adaptive stop |
|-----------|-----------|--------------|
| BTC/ETH | ~2%/day | 10% (floor) |
| Typical alt | ~4%/day | 12% |
| High-vol alt | ~7%/day | 21% |
| Extreme alt | ≥10%/day | 25% (cap) |

This prevents stops from firing on normal noise for volatile coins, while giving low-vol coins a tighter leash.

### Phase 2 — New Trade Discovery

Finds new dip-buy candidates using a multi-stage funnel before the LLM ever sees the data.

#### Drawdown circuit breaker

Before fetching any data, the bot checks unrealised portfolio P&L:
- **≥ -15% drawdown** → warn, continue with note injected into prompt
- **≥ -25% drawdown** → halt, Phase 2 exits immediately

#### Regime-adaptive rules

After fetching the Fear & Greed Index, the bot determines the current **market regime** and adjusts trading parameters. All regimes allow trading across the full 500-coin universe — there are no rank caps.

| Regime | F&G | Min conviction | Max open |
|--------|-----|:---:|:---:|
| Extreme Fear | ≤ 20 | 0.68 | 15 |
| Fear | 21–40 | 0.68 | 18 |
| Neutral | 41–60 | 0.72 | 20 |
| Greed | 61–80 | 0.75 | 20 |
| Extreme Greed | > 80 | blocked | — |

#### 7-step pipeline

1. **Fetch market data** from CoinGecko (free API, no key required):
   - Top **500 coins** by market cap with price, 1h/24h/7d change, volume
   - Trending coins
   - Fear & Greed Index (from alternative.me)

2. **Quantitative pre-screen** — scores and filters all 500 coins, returning the top 25 candidates. Hard filters run first; scoring runs after.

   **Hard filters (applied before scoring):**
   - Stablecoins and wrapped/pegged tokens (by ID list and symbol pattern)
   - Already-held coins
   - Coins already pumped past the exhaustion threshold (can't buy the top of a pump):
     - Extreme Fear: >12% in 24h
     - Fear: >18% in 24h
     - Neutral+: >30% in 24h
   - Volume below $10M 24h floor

   **Scoring factors:**

   | Factor | Range | Signal |
   |--------|-------|--------|
   | Momentum (1h×0.2 + 24h×0.5 + 7d×0.3) | ±10 pts | Price trend strength |
   | Volume spike (vol / market cap) | 0–5 pts | Unusual trading activity |
   | Relative strength vs BTC | ±5 pts | Outperforming Bitcoin |
   | Trending on CoinGecko | +3 pts | Community interest |
   | **Dip-buy bonus** (down 4–30% in 24h with volume) | up to +6 pts | Mean-reversion setup |
   | Exhaustion soft penalty (pumped but under hard-filter threshold) | up to -6 pts | Likely priced in |
   | Regime history bonus (≥65% win rate + avg 7d ≥2% in current F&G band) | +2.0 pts | Strong historical edge |
   | Regime history good (+55% win rate) | +0.75 pts | Decent historical edge |
   | Regime history penalty (≤30% win rate) | -1.5 pts | Historically poor regime |
   | Fear & Greed rank penalty (small cap in fear market) | -2 pts | Risk adjustment |

3. **Volatility + RSI enrichment** — for each of the top 25 candidates, two API calls are made:

   - **14-day price history** → computes `daily_vol_pct`, vol-adaptive stop, `stop_multiple`
   - **24h hourly chart** → computes **intraday RSI(14)** and 4h momentum

   After enrichment, two additional filters and scoring adjustments apply:

   - **RSI hard filter (dip candidates only):** if `change_24h ≤ -4%` AND `RSI > 55`, the coin is rejected — the dip has already bounced, it's too late to enter
   - **RSI scoring:**
     - RSI < 30 (deeply oversold): **+4 pts**
     - RSI 30–40 (oversold): **+2 pts**
     - RSI > 65 (overbought): **-2 pts**

   Coins are also rejected if `adaptive_stop_pct / daily_vol_pct < 1.5` (stop fires on normal noise).

4. **Fetch and filter news** — up to 60 articles from 10 RSS feeds, filtered to only articles mentioning shortlisted coins.

5. **Build LLM prompt** combining:
   - Pre-screened candidates table with score, vol, adaptive stop, **RSI**, 4h momentum, and signals
   - Filtered news headlines and summaries
   - Portfolio state (open positions, unrealised P&L, available slots)
   - **Coin behavioral profiles** — 90-day historical stats per coin
   - Recent lessons from `data/performance/lessons.json`

6. **Claude analysis** — returns trade recommendations as JSON, each with:
   - `coin_id`, `symbol`, `coin_name`
   - `conviction` — 0.0 to 1.0 confidence score
   - `reasoning` and `risks`
   - `time_horizon` — expected hold period (12h, 1d, 2d, etc. — short horizons preferred)
   - `target_pct` — typically 5–8% for dip bounces
   - `stop_loss_pct`

   Entry price sanity check: if Claude's suggested price deviates >2% from CoinGecko's live price, the trade is rejected. Entry price is always overwritten with the actual live price.

7. **Risk filters and execution**:
   - Drop trades below regime's minimum conviction threshold
   - Sector concentration cap (max 4 positions per sector)
   - Dynamic daily loss check: `max(committed_capital × 25%, $5 floor)`
   - Write approved trades to `data/trades/trades_YYYYMMDD.jsonl` and `open_positions.json`

### Phase 3 — Post-Trade Analysis (Continuous Learning)

If any positions were closed in Phase 1, the bot runs a learning cycle.

1. **Send closed trades + current positions** to Claude for review
2. **Claude extracts**: what worked, what didn't, specific actionable lessons
3. **Save lessons** to `data/performance/lessons.json`
4. **Next run** — Phase 2 injects recent lessons into the prompt

---

## Coin Behavioral Profiles

The profile pipeline (`scripts/build_coin_profiles.py`) fetches 90-day price history for up to 500 coins and computes a behavioral fingerprint for each. Profiles are saved to `data/coin_profiles/profiles.json` and injected into every Phase 2 LLM prompt.

### What's computed per coin

| Metric | Description |
|--------|-------------|
| `daily_vol_pct` | Std dev of daily log returns — how noisy is this coin? |
| `btc_corr` | Pearson correlation with BTC daily returns |
| `btc_beta` | Leverage multiple vs BTC (1.4x = amplifies BTC moves by 40%) |
| `momentum_persistence` | Lag-1 autocorrelation — positive = trending, negative = mean-reverting |
| `worst_drawdown_pct` | Worst peak-to-trough loss over 90 days |
| `regime_returns` | Avg 7-day forward return + win rate, grouped by F&G band at entry |

### Running the profile builder

```bash
# Full run — top 500 coins, 90-day history (~40 min due to rate limits)
python scripts/build_coin_profiles.py

# Smaller run for testing
python scripts/build_coin_profiles.py --coins 50 --days 30

# Update specific coins only
python scripts/build_coin_profiles.py --coin-ids bitcoin,ethereum,solana
```

Run this **weekly** to keep profiles current. Add to crontab:
```
0 6 * * 1 cd /Users/studio/Code/crypto_bot && venv/bin/python scripts/build_coin_profiles.py >> logs/intraday_cron.log 2>&1
```

---

## Risk Management

| Control | Behaviour |
|---------|-----------|
| Vol-adaptive stop-loss | Per-coin: `max(min(daily_vol × 3, 25%), 10%)` — stamped at entry |
| Trailing stop | Once in profit, close if price falls `adaptive_stop_pct` below high-water mark |
| Take-profit | Close when gain reaches `target_pct` |
| Profit protection | Peak gain ≥4%, current gain ≤50% of peak → lock in remaining profit |
| Time horizon | Close positions held past their intended duration |
| Exhaustion hard filter | Reject coins already pumped >12/18/30% in 24h (regime-dependent) |
| RSI hard filter | Reject dip candidates (down >4%) if RSI >55 (bounce already done) |
| Dynamic daily loss limit | `max(open_positions × $5 × 25%, $5)` |
| Drawdown warning | Unrealised loss ≥ -15% → note in LLM prompt |
| Drawdown halt | Unrealised loss ≥ -25% → block all new trades |
| Volatility filter | Reject coins where adaptive stop fires within 1.5 daily moves |
| Entry price sanity | Reject if LLM price deviates >2% from live CoinGecko price |
| Conviction floor | Regime-adaptive: 0.68 (Fear) → 0.75 (Greed) |
| Sector concentration | Max 4 positions per sector |

---

## Configuration

### config/config.yaml

| Setting | Default | Description |
|---------|---------|-------------|
| `trading.enabled` | `false` | Paper trading only |
| `trading.max_trades_per_run` | `5` | Max new positions per run |
| `trading.max_open_positions` | `20` | Hard cap on total open positions |
| `trading.max_position_size_usd` | `5` | Flat $5 per trade |
| `trading.min_conviction_score` | `0.7` | Fallback floor — overridden by regime at runtime |
| `trading.max_positions_per_sector` | `4` | Max positions in any single sector |
| `risk.stop_loss_percentage` | `0.20` | Fallback stop for positions without `adaptive_stop_pct` |
| `risk.max_daily_loss_pct_of_committed` | `0.25` | Daily loss limit as fraction of committed capital |
| `risk.drawdown_warning_pct` | `-15.0` | Unrealised drawdown % that triggers warning |
| `risk.drawdown_halt_pct` | `-25.0` | Unrealised drawdown % that blocks new trades |
| `data_sources.min_volume_usd` | `10000000` | $10M minimum 24h volume floor |
| `data_sources.max_candidates` | `25` | Max coins passed to LLM after pre-screen |
| `data_sources.min_stop_multiple` | `1.5` | Volatility filter threshold |
| `data_sources.coingecko.top_coins` | `500` | Universe size (auto-paginates CoinGecko) |
| `llm.model` | `claude-sonnet-4-6` | Claude model for analysis |

### Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx    # Required
```

---

## Data Files

```
data/
├── trades/
│   ├── trades_YYYYMMDD.jsonl       # All recorded trades for the day
│   ├── rejected_YYYYMMDD.jsonl     # Trades rejected by filters
│   └── summary_YYYYMMDD_HHMMSS.json
├── positions/
│   ├── open_positions.json         # Currently held positions (mutable)
│   └── resolved_trades.jsonl       # All closed trades with P&L (append-only)
├── performance/
│   └── lessons.json                # Learned lessons (injected into LLM prompts)
└── coin_profiles/
    └── profiles.json               # 90-day behavioral profiles (weekly)
```

### Open Position Record

```json
{
  "coin_id": "solana",
  "symbol": "SOL",
  "direction": "LONG",
  "entry_price": 142.50,
  "amount_invested": 5.00,
  "conviction": 0.74,
  "target_pct": 7.0,
  "time_horizon": "2d",
  "adaptive_stop_pct": 13.5,
  "screen_score": 8.2,
  "daily_vol_pct": 4.5,
  "intraday_rsi": 32.4,
  "recent_4h_pct": -2.1,
  "execution_date": "2026-04-03",
  "latest_price": 144.80,
  "pnl_pct": 1.61,
  "highest_price": 144.80
}
```

### Resolved Trade Record

```json
{
  "coin_id": "solana",
  "symbol": "SOL",
  "entry_price": 142.50,
  "close_price": 153.00,
  "pnl_pct": 7.37,
  "pnl_usd": 0.37,
  "trade_result": "CLOSED_EARLY",
  "close_type": "TAKE_PROFIT",
  "close_reason": "Price target reached: +7.4% (target was 7.0%)",
  "resolved_at": "2026-04-04T08:33:15.000000"
}
```

---

## Dashboard

Run with `streamlit run dashboard/app.py`. Opens at `http://localhost:8501` with four tabs:

- **Tab 1 — Open Positions**: Table with entry/current price, P&L, vol, adaptive stop, RSI, trailing stop
- **Tab 2 — Closed Trades**: All resolved trades filterable by result type and date
- **Tab 3 — P&L Analytics**: Cumulative P&L, daily/monthly bars, win rate, conviction scatter, distribution
- **Tab 4 — Lessons**: Timeline of lessons extracted by Claude after each close cycle

Auto-refreshes every 60 seconds.

---

## Data Sources

All free, no API keys required (except Anthropic for the LLM):

| Source | What It Provides | Rate Limits |
|--------|-----------------|-------------|
| CoinGecko API | Top 500 coins, prices, 1h/24h/7d change, volume, trending, 90-day history, 24h hourly OHLCV | ~30 req/min |
| Alternative.me | Fear & Greed Index (0–100) | Unlimited |
| 10 RSS feeds | Crypto news (CoinDesk, CoinTelegraph, Decrypt, The Block, BeInCrypto, Bitcoinist, NewsBTC, AMBCrypto, Bitcoin Magazine, CryptoPotato) | Unlimited |

---

## Running the Bot

```bash
# Full run (all 3 phases)
python scripts/intraday_trader.py

# Skip Phase 1 — only discover new trades
python scripts/intraday_trader.py --skip-closes

# Skip Phase 2 — only manage existing positions
python scripts/intraday_trader.py --skip-new-trades
```

### What to expect on first run

1. No open positions → Phase 1 prints "No open positions to check" and exits
2. Phase 2 determines market regime, fetches 500 coins across 2 CoinGecko pages
3. Quant screen scores ~150–200 qualifying coins, returns top 25
4. Volatility + RSI enrichment: one 14-day history call + one 24h hourly call per candidate (~90s due to rate limits)
5. LLM recommends dip-buy trades, phase applies risk filters
6. Total runtime: approximately **2–3 minutes** per run (RSI fetches add time vs old pipeline)

---

## Scheduling

```bash
crontab -e
```

```
# Crypto bot — every 4 hours
0 */4 * * * cd /Users/studio/Code/crypto_bot && venv/bin/python scripts/intraday_trader.py >> logs/intraday_cron.log 2>&1

# Coin profile builder — every Monday at 06:00
0 6 * * 1 cd /Users/studio/Code/crypto_bot && venv/bin/python scripts/build_coin_profiles.py >> logs/intraday_cron.log 2>&1
```

**To watch the live log:**
```bash
tail -f /Users/studio/Code/crypto_bot/logs/intraday_cron.log
```

---

## Backtesting

Measures how well the quantitative pre-screen performs by replaying screen decisions over historical CoinGecko data (no LLM involved).

```bash
# 30-day lookback, 7-day hold (default)
python scripts/backtest.py

# 60-day lookback, 3-day hold
python scripts/backtest.py --days 60 --hold 3

# Export full trade log
python scripts/backtest.py --csv results/backtest_output.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | `30` | Lookback window |
| `--hold` | `7` | Forward return window |
| `--top` | `100` | Universe size |
| `--candidates` | `15` | Max candidates per screen |
| `--volume` | `50000000` | Min 24h volume filter |
| `--stop-mult` | `1.5` | Min stop/vol ratio |
| `--step` | `7` | Days between screen runs |
| `--csv` | none | Write trade log to CSV |

---

## Project Structure

```
crypto_bot/
├── config/
│   ├── config.yaml              # Risk params, model, data sources
│   └── prompts.yaml             # LLM prompts (trade analysis, post-trade review)
├── data/                        # Runtime data (gitignored)
│   ├── trades/                  # Daily trade records and rejected trades
│   ├── positions/               # open_positions.json + resolved_trades.jsonl
│   ├── performance/             # lessons.json (continuous learning)
│   └── coin_profiles/           # profiles.json (90-day behavioral stats, weekly)
├── scripts/
│   ├── intraday_trader.py       # Main entry point — 3-phase trading loop
│   ├── build_coin_profiles.py   # Weekly pipeline: fetch history, compute profiles
│   ├── backtest.py              # Replay quant screen over historical data
│   └── helpers.py               # Position I/O, evaluation, lessons
├── src/
│   ├── data_ingestion/
│   │   ├── crypto_prices.py     # CoinGecko client — prices, trending, F&G, history, intraday RSI
│   │   └── crypto_news.py       # RSS aggregation + coin filtering (10 feeds)
│   ├── analysis/
│   │   ├── llm_analyzer.py      # Claude API wrapper with retry logic
│   │   ├── pnl_tracker.py       # P&L aggregation (daily/weekly/monthly/all-time)
│   │   ├── quant_screen.py      # Pre-screen scoring, vol/RSI enrichment, regime rules
│   │   └── coin_profiles.py     # Profile loader + LLM formatter
│   ├── trading/
│   │   ├── executor.py          # Paper trade recorder
│   │   └── risk.py              # Stop-loss, trailing stop, take-profit, profit protection, drawdown
│   └── utils/
│       ├── config.py            # YAML config with ${ENV_VAR} substitution
│       └── logger.py            # Loguru-based rotating logs
├── dashboard/
│   └── app.py                   # Streamlit dashboard (4 tabs)
├── logs/                        # Rotating daily logs (gitignored)
├── requirements.txt
├── CLAUDE.md                    # Context for Claude Code sessions
├── .env.example
└── .gitignore
```
