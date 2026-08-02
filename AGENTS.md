# crypto_bot — Agent State Document

**Read this first.** It is the current state of the project, maintained by hand.
Prefer it over re-deriving state from the repo. If you change something material,
update this file in the same commit.

Last updated: 2026-08-01 · Sibling project: `../polymarket_bot` (same redesign, see its AGENTS.md)

---

## What this is

LLM-assisted crypto trading bot. **Paper trading only** (`trading.enabled: false`).
Trades are written to files, never sent to an exchange. Runs every 4h via cron.

---

## Current state — read this before proposing changes

**A large redesign landed on 2026-08-01 (commit `df7468d`) and has NOT run live yet.**
Zero trades have executed under the new code. Any claim about whether it works is
unverified. The first real exercise is the next cron fire.

| Fact | Value |
|---|---|
| Trades executed under new code | **0** |
| Open positions | 0 |
| Resolved trades all-time | 52 |
| All-time P&L | −$0.63 |
| Last execution | 2026-07-10 |
| Tests | 88, passing |

### The finding that drove the redesign

Conviction — the score the LLM emits and the bot gated on — **carries no information
about outcomes.** Measured on the 52-trade history:

- stated 0.70 → 52.3% actual win rate
- stated 0.75 → 25.0% actual win rate
- skill score vs. base rate: **−1.47** (i.e. far worse than just knowing the base rate)

The highest-conviction bucket was the worst-performing bucket. A threshold gate on
that score was selecting against itself. Separately, `screen_score` shows no
monotonic relationship to outcomes either (62% / 31% / 50% / 50% / 75% as it rises).

**Implication for anyone working here: do not add features that consume conviction
as if it were a probability.** Route it through the calibrator.

### What is live vs. dormant

**Live on next run:**
- Volatility-targeted sizing — position size ∝ 1/daily_vol, so risk is equalised
  rather than notional. Replaces flat $5.
- Volatility-scaled triple barriers, stamped at entry.
- Re-entry cooldown (6h after a loss on the same coin).
- Premature-exit fix (see "Fixed bugs" below).
- Calibrator — fitted, but records `calibrated_prob` as a **diagnostic only**. It is
  deliberately not a gate: calibrated probabilities cluster near the base rate, so
  comparing one to a 0.68 conviction threshold would reject everything for the
  wrong reason.
- **Adaptive segment learning** — buckets outcomes by segment (sector, vol_signal,
  time_horizon, regime, entry_type), computes a Wilson interval on each, and blocks
  segments whose *upper* bound sits below break-even. Evidence decays on a 90-day
  half-life. See "Self-learning" below.

**Dormant by design:**
- **Meta-labeler classifier** — needs 100 uncensored labels, has 32. Until then the
  LLM remains the secondary model. This is the largest piece of the redesign and it
  has not started working yet.

---

## Repo structure

```
scripts/
  intraday_trader.py    Primary entry point. Three phases (below). Cron runs this.
  helpers.py            Position I/O, evaluation, time-horizon checks, path config.
  refit_models.py       Refit calibrator + secondary model; per-variant report.
  build_coin_profiles.py  Weekly job producing per-coin regime history.

src/
  data_ingestion/
    crypto_prices.py    CoinGecko client (free tier, rate-limited — expect 429s)
    crypto_news.py      RSS aggregation
  analysis/
    quant_screen.py     PRIMARY MODEL. Scores/filters candidates, sector map,
                        regime rules, volatility enrichment.
    calibration.py      Brier/log-loss/skill/reliability + PlattCalibrator.
    labeling.py         Triple-barrier labels. Marks censored exits.
    meta_labeler.py     SECONDARY MODEL. take/skip + size. LLM-backed until fitted.
    llm_analyzer.py     Claude API wrapper.
    coin_profiles.py    Historical per-coin regime stats.
    pnl_tracker.py      P&L aggregation.
  trading/
    sizing.py           Volatility-targeted position sizing.
    executor.py         Writes trades to file (paper trading).
    risk.py             Stops, trailing stops, take-profit, daily loss limits.
    (see also) analysis/adaptive.py — SegmentTracker: the self-learning gate.

config/
  config.yaml           Main config.
  variants/control.yaml A/B control arm (flat sizing, no meta-labeling).
  prompts.yaml          LLM prompts. HARD RULES section is load-bearing.

tests/                  88 tests. Run before committing.
```

### Three-phase run

1. **Position management** — price-based exits only, no LLM. Stop-loss, trailing
   stop, take-profit, profit-protection, mid-horizon stale, time-expired.
2. **New trade discovery** — screen → volatility filter → news → LLM → secondary
   model → R/R gate → conviction → sector cap → sizing → execute.
3. **Post-trade analysis** — LLM writes lessons to `lessons.json`. *See the warning
   about this under "Known-broken" below.*

---

## Commands

```bash
venv/bin/python scripts/intraday_trader.py                    # normal run
venv/bin/python scripts/intraday_trader.py --skip-new-trades  # phase 1+3 only
venv/bin/python scripts/intraday_trader.py --config config/variants/control.yaml
venv/bin/python scripts/refit_models.py --dry-run             # calibration report
venv/bin/python -m unittest discover -s tests
```

Slash command: `/analyze-performance [7d|14d|30d|YYYY-MM-DD:YYYY-MM-DD]`

---

## Key parameters (config/config.yaml)

| Key | Value | Why |
|---|---|---|
| `sizing_method` | `vol_target` | Equal risk, not equal notional |
| `target_vol_pct` | 3.0 | A full-size position targets this daily move |
| `base_position_usd` | 5.0 | Stake for a coin exactly at target vol |
| `min_conviction_score` | 0.7 | Regime rules may override upward |
| `reentry_cooldown_hours` | 6 | Blocks revenge re-entry after a loss |
| `meta_min_training_labels` | 100 | Guard against fitting on noise |
| `strategy_variant` | `vol_target_v1` | Tags trades for per-variant analysis |

---

## Fixed bugs worth remembering

**Premature exits (fixed 2026-07-22).** `check_mid_horizon_stale` and
`check_time_horizon_expired` truncated `execution_timestamp` to a date, measuring
elapsed time from *midnight* rather than the fill. A position opened at 12:17 was
treated as ~12h old at 16:00. Result: 17 trades cut at ~4h on 1-day horizons, 11.8%
win rate, the single largest P&L drag. Trades allowed to reach a barrier were 11-for-11.
Regression tests in `tests/test_helpers.py`.

**Sector map gaps (fixed 2026-08-01).** Every traded coin was falling through to
"Other", making sector concentration limits and analysis meaningless.

---

## Known-broken / do not trust

**`lessons.json` does not work — superseded by adaptive learning.** Phase 3 has an LLM write prose lessons that get
injected into the next Phase 2 prompt. Reviewing 5 sessions of history, the *same
lessons recur near-verbatim* ("don't cut early", "don't override hard rules with
regime stats") across months while the behaviour never changed. Advisory text in a
prompt is not a control. Anything that must actually happen belongs in code — this
is why the R/R gate, cooldown, and price rules are enforced mechanically.

**CoinGecko rate limits.** Free tier throttles hard. Expect `429`s and
`No price history for X — keeping with UNKNOWN volatility` in logs. Not a bug.
Unknown volatility means sizing falls back to base size rather than up-sizing.

---

## Next TODO

1. **Let the next cron fire and read the log.** Nothing is validated. First priority
   is observing, not building.
2. **Wire the control arm into cron** if running the A/B (`config/variants/control.yaml`).
   Currently written but not scheduled.
3. **Re-run `refit_models.py` as trades accumulate.** The meta-labeler wakes at 100
   uncensored labels; watch the AUC when it does.
4. **Watch the adaptive learner.** It needs ~8 effective observations per segment
   before it acts; on current history only `close_type=STALE_POSITION` clears that
   bar, and that dimension is diagnostic-only. Expect it to do nothing at first —
   that is correct.
5. **Not built, from the research:** retrieval-augmented forecasting, on-chain /
   funding-rate features, three-factor screen grounding (Liu-Tsyvinski-Wu),
   deflated Sharpe & PBO tooling for guarding against tuning overfit.

---

## Self-learning: how it works and why it is not lessons.json

`src/analysis/adaptive.py`. The design constraint came from a failure: the old
mechanism had an LLM write prose lessons into `lessons.json` for injection into the
next prompt, and the same lessons recurred near-verbatim for months while the
behaviour never changed. **Advisory text is not a control.**

The replacement is a code gate:

1. Every close is bucketed by segment and recorded with a timestamp.
2. Each segment gets a Wilson score interval on its decayed win rate.
3. A segment is blocked at entry only when its *upper* bound is below break-even —
   i.e. even an optimistic reading of the evidence says it loses.
4. Observations decay (90-day half-life), so suppression lifts as evidence ages and
   the system tracks a changing market rather than last quarter's.

Deliberate properties:
- **Restraint over reactivity.** Three losses in a row does nothing. `min_samples`
  is in *effective* (decayed) observations, and the z is 1.28 (~80%) — high enough
  to ignore noise, low enough that it will eventually act.
- **No leakage.** `GATING_DIMENSIONS` are all knowable at entry.
  `DIAGNOSTIC_DIMENSIONS` (close_type etc.) are reportable but `evaluate_trade()`
  **raises** if you try to gate on them.
- **Per-dimension, not combined.** Combined keys fragment the sample so finely that
  nothing reaches significance.

Known limitation, worth fixing later: **suppression is win-rate based, so it cannot
see P&L-shaped failures.** In the sibling repo, `position=NO` wins 58% of the time
while losing $28 — a segment that wins often but loses big is invisible to this
gate. An expectancy-based criterion would catch it.

Inspect with `venv/bin/python scripts/refit_models.py --dry-run`.

## Working agreements

- **Never** claim a change works without running it. Say what was verified and what wasn't.
- Enforce rules in code, not prompts, whenever the rule must actually hold.
- Guard every fitted model with a minimum sample size; an undersized fit is worse
  than none because it looks confident.
- Report Brier and skill score, not win rate. Win rate is what let a 51%-accurate
  signal look healthy.
- Keep both bots on paper trading well past the point the metrics look good.
