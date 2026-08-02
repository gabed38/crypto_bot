"""
Crypto Trading Bot Dashboard

Run with: venv/bin/streamlit run dashboard/app.py

Rebuilt 2026-08-01 for the calibration/meta-labeling redesign.

Two deliberate changes from the previous version:

  1. Win rate is no longer the headline. It is the metric that let a signal with a
     -1.47 skill score look healthy. Brier and skill score lead instead; win rate is
     still shown, demoted, because it is intuitive — not because it is informative.

  2. Everything is filterable by strategy_variant. Without that, parallel A/B arms
     average together in one view and the experiment is invisible.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.pnl_tracker import PnLTracker, _amount_invested, _pnl_usd
from src.analysis.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    reference_brier,
    reliability_curve,
    skill_score,
)
from src.analysis.labeling import label_closed_trade
from src.analysis.meta_labeler import MetaLabeler
from src.analysis.adaptive import (
    DIAGNOSTIC_DIMENSIONS,
    GATING_DIMENSIONS,
    SegmentTracker,
)

OPEN_POSITIONS_FILE = ROOT / "data/positions/open_positions.json"
RESOLVED_FILE = ROOT / "data/positions/resolved_trades.jsonl"
LESSONS_FILE = ROOT / "data/performance/lessons.json"

GOOD, BAD, WARN, MUTED = "#2f7d5a", "#a94442", "#c98a2e", "#8f90a6"

st.set_page_config(page_title="Crypto Bot", page_icon="₿", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stDataFrame { font-size: 13px; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
  .note { color: #8f90a6; font-size: 0.86rem; }
</style>
""", unsafe_allow_html=True)


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_open_positions() -> List[Dict]:
    if not OPEN_POSITIONS_FILE.exists():
        return []
    try:
        with open(OPEN_POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_resolved_trades() -> List[Dict]:
    trades = PnLTracker(RESOLVED_FILE).load_resolved()
    # Stamp triple-barrier labels on older rows that predate the labeler, so
    # censored early exits are excluded from scoring rather than counted as real.
    for t in trades:
        if "tb_label" not in t:
            t.update(label_closed_trade(t))
    return trades


@st.cache_data(ttl=60)
def load_lessons() -> List:
    if not LESSONS_FILE.exists():
        return []
    try:
        with open(LESSONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def variants_in(trades: List[Dict]) -> List[str]:
    return sorted({t.get("strategy_variant") or "untagged" for t in trades})


def tb_outcome(trade: Dict) -> Optional[int]:
    """Binary outcome for scoring: did the profit barrier get hit first?"""
    lab = trade.get("tb_label")
    return None if lab is None else (1 if lab == 1 else 0)


# ── sidebar: variant filter ───────────────────────────────────────────────────

all_resolved = load_resolved_trades()
all_open = load_open_positions()

st.sidebar.header("Filters")
variant_options = ["All variants"] + variants_in(all_resolved + all_open)
chosen = st.sidebar.selectbox(
    "Strategy variant", variant_options,
    help="Parallel A/B arms write to isolated state. Compare them here rather than "
         "averaging them together.",
)
if chosen != "All variants":
    resolved = [t for t in all_resolved if (t.get("strategy_variant") or "untagged") == chosen]
    positions = [p for p in all_open if (p.get("strategy_variant") or "untagged") == chosen]
else:
    resolved, positions = all_resolved, all_open

exclude_censored = st.sidebar.checkbox(
    "Exclude censored exits from scoring", value=True,
    help="Discretionary early exits (STALE_POSITION, STRATEGY_RESET) describe the exit "
         "logic in force at the time, not whether the entry had merit. 20 of the first "
         "52 trades were censored artifacts of the premature-exit bug.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span class='note'>{len(all_resolved)} resolved · {len(all_open)} open<br>"
    f"Refreshed {datetime.now().strftime('%H:%M:%S')}</span>",
    unsafe_allow_html=True,
)

st.title("₿ Crypto Trading Bot")
st.caption(f"Variant: **{chosen}** · paper trading only")

if not all_resolved and not all_open:
    st.info("No trades yet. The redesign has not executed a trade — the next cron run "
            "is the first one under the new code.")

tab_scoring, tab_models, tab_open, tab_resolved, tab_pnl, tab_lessons = st.tabs(
    ["Scoring", "Models & Learning", "Open", "Resolved", "P&L", "Lessons"]
)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — the headline. Brier and skill, not win rate.
# ══════════════════════════════════════════════════════════════════════════════

with tab_scoring:
    scored = [t for t in resolved
              if tb_outcome(t) is not None and t.get("conviction") is not None]
    if exclude_censored:
        scored = [t for t in scored if not t.get("tb_censored", False)]

    if not scored:
        st.info("Nothing scoreable yet. Scoring needs a conviction score and a "
                "triple-barrier outcome on the same trade.")
    else:
        probs = [float(t["conviction"]) for t in scored]
        outs = [tb_outcome(t) for t in scored]

        bs = brier_score(probs, outs)
        ref = reference_brier(outs)
        sk = skill_score(probs, outs)
        ece = expected_calibration_error(probs, outs)
        wins = sum(outs)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Brier", f"{bs:.3f}", help="Lower is better. 0.25 = always guessing 50%.")
        c2.metric("Base-rate Brier", f"{ref:.3f}",
                  help="What you'd score knowing only how often trades win. The number to beat.")
        c3.metric("Skill score", f"{sk:+.3f}",
                  delta="beats base rate" if sk and sk > 0 else "worse than base rate",
                  delta_color="normal" if sk and sk > 0 else "inverse",
                  help="1 - (model Brier / base-rate Brier). Negative means the score is "
                       "actively worse than knowing nothing.")
        c4.metric("Calibration error", f"{ece:.3f}",
                  help="Mean gap between stated confidence and realised frequency.")
        c5.metric("Win rate", f"{100 * wins / len(outs):.0f}%",
                  help="Shown because it's intuitive, not because it's informative — "
                       "this is the metric that made a -1.47 skill score look healthy.")

        if sk is not None and sk < 0:
            st.warning(
                f"**Skill score is negative ({sk:+.3f}).** The conviction score is doing "
                f"worse than simply knowing the base rate. Do not gate or size on it "
                f"directly — route it through the calibrator.", icon="⚠️")

        st.subheader("Reliability — stated vs. delivered")
        curve = reliability_curve(probs, outs, n_bins=6)
        if curve:
            cdf = pd.DataFrame(curve)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color=MUTED),
                                     name="perfect calibration"))
            fig.add_trace(go.Scatter(
                x=cdf["mean_forecast"], y=cdf["observed_freq"], mode="markers+lines",
                marker=dict(size=cdf["n"].clip(6, 28), color=GOOD), name="actual",
                hovertemplate="stated %{x:.2f}<br>actual %{y:.2f}<extra></extra>"))
            fig.update_layout(height=380, xaxis_title="stated probability",
                              yaxis_title="observed frequency",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                cdf[["bin_low", "bin_high", "n", "mean_forecast", "observed_freq", "gap"]],
                use_container_width=True, hide_index=True)
            st.caption("Points below the dashed line mean overconfidence — the model "
                       "claimed more than it delivered.")


# ══════════════════════════════════════════════════════════════════════════════
# MODELS & LEARNING
# ══════════════════════════════════════════════════════════════════════════════

with tab_models:
    variant_key = chosen if chosen != "All variants" else "default"

    st.subheader("Calibrator")
    cal = PlattCalibrator.load(variant=variant_key)
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "fitted" if cal.fitted else "not fitted")
    c2.metric("Samples", cal.n_samples)
    c3.metric("Method", PlattCalibrator.recommend_method(cal.n_samples).split("—")[0].strip())
    if cal.fitted:
        m = cal.fit_metrics or {}
        mc1, mc2, mc3 = st.columns(3)
        if m.get("brier_raw") is not None:
            mc1.metric("Brier raw → calibrated",
                       f"{m['brier_raw']:.3f} → {m['brier_calibrated']:.3f}")
        if m.get("ece_raw") is not None:
            mc2.metric("ECE raw → calibrated", f"{m['ece_raw']:.3f} → {m['ece_calibrated']:.3f}")
        if m.get("skill_calibrated") is not None:
            mc3.metric("Skill after calibration", f"{m['skill_calibrated']:+.4f}")
        pts = [0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
        mapping = pd.DataFrame({"conviction": pts,
                                "calibrated probability": [round(cal.transform(p), 3) for p in pts]})
        st.dataframe(mapping, use_container_width=True, hide_index=True)
        spread = mapping["calibrated probability"].max() - mapping["calibrated probability"].min()
        if spread < 0.10:
            st.info(
                f"The calibrator compresses the whole conviction range into a "
                f"{spread:.2f}-wide band around the base rate. That is the calibrator "
                f"concluding the score carries almost no information — not a bug.", icon="ℹ️")
    else:
        st.caption(f"Unfitted — raw conviction passes through unchanged. "
                   f"Needs {PlattCalibrator.MIN_SAMPLES} samples.")

    st.markdown("---")
    st.subheader("Secondary model (meta-labeling)")
    meta = MetaLabeler.load(variant=variant_key)
    trainable = sum(1 for t in resolved if not t.get("tb_censored", True))
    c1, c2, c3 = st.columns(3)
    c1.metric("Backend", meta.backend)
    c2.metric("Uncensored labels", f"{trainable} / {meta.min_training_labels}")
    c3.metric("AUC", f"{meta.fit_metrics.get('auc'):.3f}"
              if meta.fitted and meta.fit_metrics.get("auc") else "—")
    if not meta.fitted:
        st.caption(
            f"Dormant by design. The LLM remains the secondary model until "
            f"{meta.min_training_labels} uncensored labels exist — a classifier fitted "
            f"on fewer is worse than none, because it looks confident.")
    if meta.fitted and meta.weights:
        w = pd.DataFrame({"feature": list(meta.weights), "weight": list(meta.weights.values())})
        w = w.reindex(w["weight"].abs().sort_values(ascending=False).index)
        st.plotly_chart(
            px.bar(w, x="weight", y="feature", orientation="h", height=300, color="weight",
                   color_continuous_scale=[BAD, MUTED, GOOD]),
            use_container_width=True)

    st.markdown("---")
    st.subheader("Adaptive segment learning")
    st.caption("Blocks segments only when the UPPER bound of their win-rate interval "
               "sits below break-even. A losing streak alone will not trigger it.")
    tracker = SegmentTracker.rebuild_from_history(
        resolved, GATING_DIMENSIONS + DIAGNOSTIC_DIMENSIONS,
        outcome_fn=lambda t: 1 if (t.get("pnl_pct") or 0) > 0 else 0,
        half_life_days=90.0, min_samples=8.0,
    )
    rows = tracker.report(break_even=0.5)
    if not rows:
        st.info("No segment data yet.")
    else:
        sdf = pd.DataFrame(rows)
        sdf["gating"] = ~sdf["segment"].str.startswith(
            tuple(d + "=" for d in DIAGNOSTIC_DIMENSIONS))
        sdf["status"] = sdf.apply(
            lambda r: "BLOCKED" if r["suppressed"] and r["gating"]
            else ("blocked (diagnostic only)" if r["suppressed"] else ""), axis=1)
        blocked = sdf[sdf["suppressed"] & sdf["gating"]]
        if len(blocked):
            st.error(f"{len(blocked)} segment(s) currently blocked at entry: "
                     + ", ".join(blocked["segment"]), icon="🚫")
        else:
            st.success("No segments blocked — either performance is acceptable or "
                       "evidence is still too thin to act on.", icon="✅")
        st.dataframe(
            sdf[["segment", "n_effective", "win_rate", "ci_high", "total_pnl", "status"]]
            .rename(columns={"n_effective": "n (decayed)", "win_rate": "win rate",
                             "ci_high": "upper bound", "total_pnl": "P&L"}),
            use_container_width=True, hide_index=True)
        st.caption("Win-rate based, so it cannot see P&L-shaped failures — a segment "
                   "that wins often but loses big stays invisible. Check the P&L column.")


# ══════════════════════════════════════════════════════════════════════════════
# OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_open:
    if not positions:
        st.info("No open positions.")
    else:
        sizes = [_amount_invested(p) for p in positions]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open", len(positions))
        c2.metric("Capital at risk", f"${sum(sizes):,.2f}")
        c3.metric("Unrealised P&L", f"${sum(p.get('pnl_usd') or 0 for p in positions):+,.2f}")
        c4.metric("Size range", f"${min(sizes):.2f}–${max(sizes):.2f}",
                  help="Under vol-targeting these should differ — equal risk, not equal notional.")

        st.dataframe(pd.DataFrame([{
            "Symbol": (p.get("symbol") or "?").upper(),
            "Sector": p.get("sector", "—"),
            "Entry": p.get("entry_price"),
            "Latest": p.get("latest_price"),
            "P&L %": p.get("pnl_pct"),
            "Invested": _amount_invested(p),
            "Size×": p.get("sizing_scale_factor"),
            "Daily vol %": p.get("sizing_daily_vol_pct") or p.get("daily_vol_pct"),
            "TP %": p.get("profit_barrier_pct"),
            "SL %": p.get("stop_barrier_pct"),
            "Conviction": p.get("conviction"),
            "Calibrated": p.get("calibrated_prob"),
            "Meta": p.get("meta_decision"),
            "Horizon": p.get("time_horizon"),
            "Variant": p.get("strategy_variant", "untagged"),
        } for p in positions]), use_container_width=True, hide_index=True)

        if any(p.get("sizing_scale_factor") for p in positions):
            st.subheader("Risk equalisation check")
            chk = pd.DataFrame([{
                "symbol": (p.get("symbol") or "?").upper(),
                "risk (vol × size)": (p.get("sizing_daily_vol_pct") or 0)
                                     * _amount_invested(p) / 100,
            } for p in positions])
            st.plotly_chart(px.bar(chk, x="symbol", y="risk (vol × size)", height=280),
                            use_container_width=True)
            st.caption("Bars should be roughly level — that is what vol-targeting buys you.")


# ══════════════════════════════════════════════════════════════════════════════
# RESOLVED
# ══════════════════════════════════════════════════════════════════════════════

with tab_resolved:
    if not resolved:
        st.info("No resolved trades.")
    else:
        censored = sum(1 for t in resolved if t.get("tb_censored"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resolved", len(resolved))
        c2.metric("Uncensored", len(resolved) - censored,
                  help="Censored = discretionary early exit, not a real barrier outcome.")
        c3.metric("Realised P&L", f"${sum(t.get('pnl_usd') or 0 for t in resolved):+,.2f}")
        c4.metric("Censored", censored)

        st.subheader("Exit type breakdown")
        ex: Dict[str, Dict] = {}
        for t in resolved:
            e = ex.setdefault(t.get("close_type", "?"), {"n": 0, "wins": 0, "pnl": 0.0})
            e["n"] += 1
            e["wins"] += 1 if (t.get("pnl_pct") or 0) > 0 else 0
            e["pnl"] += t.get("pnl_usd") or 0
        st.dataframe(pd.DataFrame([
            {"Exit": k, "n": v["n"], "Win %": round(100 * v["wins"] / v["n"]),
             "P&L": round(v["pnl"], 2)} for k, v in ex.items()
        ]).sort_values("n", ascending=False), use_container_width=True, hide_index=True)

        st.dataframe(pd.DataFrame([{
            "Resolved": (t.get("resolved_at") or "")[:16].replace("T", " "),
            "Symbol": (t.get("symbol") or "?").upper(),
            "Exit": t.get("close_type"),
            "TB": {1: "profit", -1: "stop", 0: "neutral"}.get(t.get("tb_label"), "—"),
            "Censored": "yes" if t.get("tb_censored") else "",
            "P&L %": t.get("pnl_pct"),
            "P&L $": t.get("pnl_usd"),
            "Invested": t.get("amount_invested"),
            "Conviction": t.get("conviction"),
            "Screen": t.get("screen_score"),
            "Variant": t.get("strategy_variant", "untagged"),
        } for t in sorted(resolved, key=lambda x: x.get("resolved_at", ""), reverse=True)]),
            use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# P&L
# ══════════════════════════════════════════════════════════════════════════════

with tab_pnl:
    if not resolved:
        st.info("No resolved trades.")
    else:
        running = 0.0
        cum = []
        for t in sorted(resolved, key=lambda x: x.get("resolved_at", "")):
            running += t.get("pnl_usd") or 0
            cum.append({"resolved_at": (t.get("resolved_at") or "")[:10],
                        "cumulative": round(running, 2),
                        "variant": t.get("strategy_variant", "untagged")})
        cdf = pd.DataFrame(cum)

        st.subheader("Cumulative P&L")
        if chosen == "All variants" and cdf["variant"].nunique() > 1:
            fig = px.line(cdf, x="resolved_at", y="cumulative", color="variant", height=380)
            st.caption("Split by variant — this is the A/B comparison.")
        else:
            fig = px.line(cdf, x="resolved_at", y="cumulative", height=380)
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        st.plotly_chart(fig, use_container_width=True)

        if len(variants_in(resolved)) > 1:
            st.subheader("Variant comparison")
            comp = []
            for v in variants_in(resolved):
                vt = [t for t in resolved if (t.get("strategy_variant") or "untagged") == v]
                sc = [t for t in vt
                      if tb_outcome(t) is not None and t.get("conviction") is not None]
                p = [float(t["conviction"]) for t in sc]
                o = [tb_outcome(t) for t in sc]
                comp.append({
                    "Variant": v, "Trades": len(vt),
                    "P&L": round(sum(t.get("pnl_usd") or 0 for t in vt), 2),
                    "Brier": round(brier_score(p, o), 4) if sc else None,
                    "Skill": round(skill_score(p, o), 4) if sc else None,
                })
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

        st.subheader("P&L by sector")
        sec: Dict[str, List[float]] = {}
        for t in resolved:
            sec.setdefault(t.get("sector", "Other"), []).append(t.get("pnl_usd") or 0)
        sdf = pd.DataFrame([{"sector": k, "P&L": round(sum(v), 2), "n": len(v)}
                            for k, v in sec.items()]).sort_values("P&L")
        st.plotly_chart(px.bar(sdf, x="sector", y="P&L", color="P&L", height=320,
                               color_continuous_scale=[BAD, MUTED, GOOD]),
                        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# LESSONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_lessons:
    st.warning(
        "**These do not drive behaviour.** Reviewing months of history, the same "
        "lessons recur near-verbatim while what they warned about kept happening — "
        "advisory text in a prompt is not a control. The adaptive segment learner "
        "under *Models & Learning* is the mechanism that actually changes what the "
        "bot does. Keep these as commentary.", icon="⚠️")
    lessons = load_lessons()
    if not lessons:
        st.info("No lessons recorded.")
    else:
        for entry in reversed(lessons[-12:]):
            with st.expander(f"{entry.get('date','?')} — "
                             f"{entry.get('resolved_count',0)} resolved, "
                             f"P&L ${entry.get('pnl_usd',0):+.2f}"):
                for label, key in (("What worked", "what_worked"),
                                   ("What didn't", "what_didnt_work"),
                                   ("Lessons", "lessons")):
                    items = entry.get(key) or []
                    if items:
                        st.markdown(f"**{label}**")
                        for it in items:
                            st.markdown(f"- {it}")
