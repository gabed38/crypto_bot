# Crypto Trading Bot — Claude Context

## What this project is

A crypto trading bot that uses Claude (Anthropic API) to analyze crypto markets and recommend trades. **Paper trading only** (`trading.enabled: false` in `config/config.yaml`). All trades are written to files, not executed on any exchange.

## Slash commands

```bash
/analyze-performance        # Last 30 days
/analyze-performance 7d     # Last 7 days
/analyze-performance 14d    # Last 14 days
/analyze-performance 2026-04-01:2026-05-01  # Explicit date range
```

Reads `resolved_trades.jsonl`, `rejected_YYYYMMDD.jsonl`, `lessons.json`, and `open_positions.json` to produce a structured performance report with win rate, P&L breakdown, exit type analysis, sector performance, rejection patterns, recurring lesson themes, and specific strategy tweak recommendations.

## Running the bot

```bash
# Primary script (always use this one)
venv/bin/python scripts/intraday_trader.py

# Skip position management (Phase 1)
venv/bin/python scripts/intraday_trader.py --skip-closes

# Skip new trade discovery (Phase 2)
venv/bin/python scripts/intraday_trader.py --skip-new-trades

# Dashboard
venv/bin/streamlit run dashboard/app.py
```

## Environment

- API keys live in `.env` (not committed) — loaded via `python-dotenv`
- Required: `ANTHROPIC_API_KEY`
- Config uses `${ENV_VAR}` substitution via `config/config.yaml`

## Project structure

```
scripts/
  intraday_trader.py         # Primary script — runs every 4 hours via cron
  helpers.py                 # Shared helpers — position I/O, evaluation, lessons

src/
  data_ingestion/
    crypto_prices.py         # CoinGecko API client (free, no key needed)
    crypto_news.py           # RSS feeds news aggregation
  analysis/
    llm_analyzer.py          # Claude API wrapper
    pnl_tracker.py           # P&L aggregation across resolved trades
  trading/
    executor.py              # Writes trades to file (paper trading)
    risk.py                  # Stop-loss and daily loss limits
  utils/
    config.py                # YAML config with env var substitution
    logger.py                # Loguru-based logging

config/
  config.yaml                # Main config (API keys, risk params, model)
  prompts.yaml               # LLM prompts for trade analysis and performance review

dashboard/
  app.py                     # Streamlit dashboard
```

## Intraday workflow (every 4 hours)

### Phase 1 — Position Management
1. Load all open positions from `data/positions/open_positions.json`
2. Batch fetch current prices from CoinGecko
3. Positions down 15%+ → close as STOP_LOSS immediately
4. Run LLM hold/close analysis on remaining → close TAKE_PROFIT / CUT_LOSS
5. Save updated open positions

### Phase 2 — New Trade Discovery
1. Drawdown circuit breaker: halt if unrealised portfolio loss > 25%
2. Fetch top 50 coins (prices, volume, market cap), trending coins, Fear & Greed Index
3. Quantitative pre-screen: score on momentum / volume spike / RS vs BTC → top 15 candidates
4. Volatility filter: fetch 14-day price history, reject if stop-multiple < 1.5
5. Fetch news filtered to shortlisted coins only
6. Send screened data + news + portfolio state + lessons to Claude
7. Filter: drop below `min_conviction_score` (0.7); cap: max 5 per run, max 20 open

### Phase 3 — Post-Trade Analysis
1. If any positions closed in Phase 1, run LLM review
2. Extract lessons and save to `data/performance/lessons.json`
3. Lessons are injected into next run's Phase 2 prompt (continuous learning)

## Data sources (all free, no API keys required)

- **CoinGecko API** — coin prices, market cap, volume, trending coins (free tier)
- **Alternative.me** — Fear & Greed Index (free)
- **RSS Feeds** — CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine

## Key differences from polymarket_bot

- No market classification by niche — purely crypto
- Positions are LONG on specific coins (not YES/NO binary outcomes)
- P&L is based on price movement (not binary resolution)
- Positions don't "resolve" naturally — bot decides when to close
- Stop-loss at 15% down triggers automatic close
- CoinGecko coin IDs used as position identifiers (e.g., "bitcoin", "ethereum")

## Risk management

- Max 20 open positions, max 5 new trades per run
- $5 flat per trade
- 15% stop-loss (automatic)
- Min 0.7 conviction to trade
- Dynamic daily loss limit: `max(committed_capital × 25%, $5 floor)` — scales with exposure
- Drawdown circuit breaker: warn at -15% unrealised, halt new trades at -25% unrealised
- Volatility filter: coins where stop-loss fires within 1.5 average daily moves are rejected

## LLM model

Currently `claude-sonnet-4-6` in `config/config.yaml`. Update there if needed.
