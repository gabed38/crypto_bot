# Performance Analysis

Analyze the crypto trading bot's historical performance over a specified time window and produce a structured report with actionable strategy insights.

## Arguments

The user may pass a time window as `$ARGUMENTS`:
- `7d`, `14d`, `30d`, `90d` — last N days relative to today
- `2026-04-01:2026-05-01` — explicit date range (inclusive)
- No argument — default to last 30 days

## Data files to read

All paths are relative to the repo root.

| File | Format | What it contains |
|------|--------|-----------------|
| `data/positions/resolved_trades.jsonl` | One JSON object per line | Every closed trade with entry/exit prices, P&L, close type, sector, conviction, screen_score |
| `data/positions/open_positions.json` | JSON array | Currently open positions with unrealised P&L |
| `data/trades/rejected_YYYYMMDD.jsonl` | Multiple JSON objects per file (not JSONL — parse with a streaming decoder) | Trades rejected by filters, each with a `rejection_reason` field |
| `data/performance/lessons.json` | JSON array | Post-trade lesson sessions, each with `date`, `win_rate_pct`, `pnl_usd`, `lessons[]`, `what_worked[]`, `what_didnt_work[]`, `reasoning_quality` |

## Parsing notes

**`resolved_trades.jsonl`** — standard JSONL, one object per line. Filter on `resolved_at` (ISO datetime string) for the time window.

**`rejected_YYYYMMDD.jsonl`** — each file is NOT line-delimited; it contains multiple JSON objects concatenated. Use a streaming JSON decoder (e.g. Python's `json.JSONDecoder().raw_decode()` in a loop) or `grep -l` to find the right date files. Filter files by filename date (YYYYMMDD) to match the time window.

**`open_positions.json`** — always include this regardless of time window; it shows the current state.

## Steps

### 1. Determine the date range

Parse `$ARGUMENTS` to get `start_date` and `end_date`. If no argument, use the last 30 days. Print the range you're analyzing at the top of the report.

### 2. Load resolved trades in range

Read `data/positions/resolved_trades.jsonl` line by line. Keep trades where `resolved_at` falls within the range. Extract:
- `symbol`, `coin_id`, `sector`
- `entry_price`, `close_price`, `pnl_pct`, `pnl_usd`
- `close_type` — TAKE_PROFIT, STOP_LOSS, TRAILING_STOP, PROFIT_PROTECTION, TIME_EXPIRED, CUT_LOSS, STRATEGY_RESET
- `conviction`, `screen_score`, `daily_vol_pct`, `stop_multiple`, `adaptive_stop_pct`
- `time_horizon`, `execution_date`, `resolved_at`
- `reasoning` (first 120 chars), `close_reason`

### 3. Load rejected trades in range

Find all `data/trades/rejected_YYYYMMDD.jsonl` files whose date falls in the range. For each file, stream-parse the concatenated JSON objects. Extract `symbol`, `coin_id`, `conviction`, `sector`, `rejection_reason`, and the date from the filename.

### 4. Load lessons in range

Read `data/performance/lessons.json`. Filter sessions where `date` falls in range. Collect `lessons[]` across all matching sessions.

### 5. Load open positions

Read `data/positions/open_positions.json` for the current portfolio snapshot. Compute unrealised P&L totals.

---

## Report structure

Output the full report in this exact order. Use markdown headers so it's easy to scan.

---

### `## Performance Summary — [DATE RANGE]`

One-line snapshot table:

| Metric | Value |
|--------|-------|
| Period | start → end |
| Trades closed | N |
| Win rate | X% (wins / total) |
| Realised P&L | $X.XX |
| Avg P&L per trade | $X.XX |
| Best trade | SYMBOL +X% ($X.XX) |
| Worst trade | SYMBOL −X% ($X.XX) |
| Open positions | N (unrealised P&L: $X.XX) |
| Rejection rate | N rejected / N total considered |

---

### `## Exit Type Breakdown`

Table showing how positions were closed. For each `close_type`, show count, win rate (% with positive pnl_pct), and total P&L.

| Exit type | Count | Win rate | Total P&L |
|-----------|-------|----------|-----------|
| TAKE_PROFIT | … | 100% | … |
| TRAILING_STOP | … | … | … |
| PROFIT_PROTECTION | … | … | … |
| STOP_LOSS | … | … | … |
| TIME_EXPIRED | … | … | … |
| CUT_LOSS | … | … | … |

Note if TIME_EXPIRED or CUT_LOSS trades are clustering — that suggests time horizons are wrong or entry quality is slipping.

---

### `## Sector Performance`

Group closed trades by `sector`. For each sector: trade count, win rate, total P&L, avg conviction.

Highlight any sector where win rate < 40% or where the bot is repeatedly losing — that's a signal to reduce or avoid.

---

### `## Conviction & Screen Score Analysis`

Split trades into win / loss buckets and show:
- Average conviction for wins vs losses
- Average screen_score for wins vs losses
- If losses have higher conviction than wins, flag it — the bot is overconfident on bad trades
- If screen_score doesn't correlate with outcome, call that out too

Optionally: if there are ≥5 trades, show a simple breakdown by conviction band (e.g. 0.65–0.70, 0.70–0.75, 0.75+).

---

### `## Rejection Patterns`

Tally `rejection_reason` across all rejected files in the window. Show a frequency table:

| Rejection reason | Count |
|-----------------|-------|
| Below min conviction (0.85) | … |
| Open position cap reached | … |
| … | … |

Then: what fraction of total candidate trades (rejected + executed) actually got through? A very high rejection rate might mean the filters are too tight; a very low rate might mean the bot isn't being selective enough.

---

### `## Lessons Extracted This Period`

List the recurring themes from `lessons.json` for the period. Look for lessons that appear more than once — repetition means the bot keeps making the same mistake.

Group related lessons together under a theme label (e.g. "Entry timing", "Position sizing", "Exit discipline", "Regime rules"). Don't paste all the raw text — synthesise the key patterns.

---

### `## Strategy Observations`

This is the most important section. Based on everything above, identify **3–5 concrete, actionable findings** about what to tweak. Frame each one as:

**Finding:** what the data shows  
**Evidence:** the specific numbers or pattern  
**Suggested tweak:** one specific change to the strategy, risk params, or prompt

Example format:
> **Finding:** TIME_EXPIRED trades are dragging down P&L  
> **Evidence:** 8 of 12 losses closed via TIME_EXPIRED, avg P&L −2.1%  
> **Suggested tweak:** Reduce time horizon from 2d to 1d for dip-buy entries, or add a rule to cut positions at breakeven when time horizon is 50% elapsed and P&L < +1%

---

### `## Open Positions Snapshot`

Table of current open positions for context:

| Symbol | Entry | Current | P&L% | Horizon | Days held | Conviction |
|--------|-------|---------|-------|---------|-----------|------------|

Flag any positions that are past their time horizon or sitting at a large loss — these need attention.

---

## Tone and depth

- Be direct. The user is trying to find what's broken and fix it.
- If a pattern is clearly bad (e.g., 0% win rate on a close type, repeated same lesson), say so plainly.
- If the sample size is too small for a given slice (< 5 trades), note the caveat instead of drawing strong conclusions.
- Don't pad the report with generic crypto commentary. Every sentence should be grounded in the actual data you read.
