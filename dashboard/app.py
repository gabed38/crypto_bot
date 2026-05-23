"""
Crypto Trading Bot Dashboard
Run with: streamlit run dashboard/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.pnl_tracker import PnLTracker, _amount_invested, _pnl_usd

# ── data paths ────────────────────────────────────────────────────────────────
OPEN_POSITIONS_FILE = ROOT / "data/positions/open_positions.json"
RESOLVED_FILE = ROOT / "data/positions/resolved_trades.jsonl"
LESSONS_FILE = ROOT / "data/performance/lessons.json"

DIRECTION_COLORS = {"LONG": "#00A86B", "SHORT": "#FF6B6B"}
RESULT_COLORS = {"WIN": "#00A86B", "LOSS": "#FF6B6B", "CLOSED_EARLY": "#FFA500", "UNREALIZED": "#888888"}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Bot Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 8px; padding: 12px 16px; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ── data loaders ──────────────────────────────────────────────────────────────

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
    tracker = PnLTracker(RESOLVED_FILE)
    return tracker.load_resolved()


@st.cache_data(ttl=60)
def load_lessons() -> List:
    if not LESSONS_FILE.exists():
        return []
    try:
        with open(LESSONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


VOL_SIGNAL_COLORS = {
    "LOW":     "#00A86B",
    "MEDIUM":  "#F7931A",
    "HIGH":    "#FF6B6B",
    "EXTREME": "#CC0000",
    "UNKNOWN": "#888888",
}

def _vol_badge(signal: str) -> str:
    """Return a coloured HTML badge for a vol signal."""
    color = VOL_SIGNAL_COLORS.get(signal, "#888888")
    return f'<span style="background:{color};color:#fff;padding:2px 6px;border-radius:4px;font-size:0.75em">{signal}</span>'


def open_positions_df(positions: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in positions:
        entry = p.get("entry_price") or 0
        latest = p.get("latest_price") or entry
        unrealized_pct = p.get("pnl_pct")
        unrealized_usd = p.get("pnl_usd")

        if unrealized_pct is None and entry and latest and entry > 0:
            direction = p.get("direction", "LONG")
            if direction == "LONG":
                unrealized_pct = round((latest - entry) / entry * 100, 1)
            else:
                unrealized_pct = round((entry - latest) / entry * 100, 1)
            amt = p.get("amount_invested") or 5.0
            unrealized_usd = round((unrealized_pct / 100) * amt, 2)

        # Trailing stop info
        trailing_stop = p.get("trailing_stop_price")
        hwm = p.get("highest_price") or p.get("lowest_price")

        rows.append({
            "Coin": f"{p.get('symbol', '?').upper()} ({p.get('coin_name', '')})",
            "Sector": p.get("sector", "Other"),
            "Direction": p.get("direction", "LONG"),
            "Entry": entry,
            "Now": latest,
            "Unrlzd %": unrealized_pct,
            "Unrlzd $": unrealized_usd,
            "Trailing Stop": trailing_stop,
            "HWM": hwm,
            "Vol/day": p.get("daily_vol_pct"),
            "Stop×": p.get("stop_multiple"),
            "Vol Risk": p.get("vol_signal", "UNKNOWN"),
            "Invested": p.get("amount_invested") or 5.0,
            "Conviction": p.get("conviction"),
            "Horizon": p.get("time_horizon", "?"),
            "Open Since": p.get("execution_date", "?"),
        })
    return pd.DataFrame(rows)


def resolved_trades_df(trades: List[Dict]) -> pd.DataFrame:
    rows = []
    for t in trades:
        ts = t.get("resolved_at") or t.get("executed_at") or ""
        date = ts[:10] if ts else "?"
        rows.append({
            "Date": date,
            "Coin": f"{t.get('symbol', '?').upper()}",
            "Direction": t.get("direction", "LONG"),
            "Result": t.get("trade_result", "?"),
            "Close Type": t.get("close_type", ""),
            "Entry": t.get("entry_price"),
            "Exit": t.get("close_price") or t.get("latest_price"),
            "P&L %": t.get("pnl_pct"),
            "P&L $": t.get("pnl_usd"),
            "Invested": t.get("amount_invested"),
            "Conviction": t.get("conviction"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date", ascending=False)
    return df


def color_pnl(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "color: #00A86B" if float(val) >= 0 else "color: #FF6B6B"


def style_resolved_df(df: pd.DataFrame):
    return (
        df.style
        .applymap(color_pnl, subset=["P&L %", "P&L $"])
        .format({
            "Entry": lambda v: f"${v:,.4f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
            "Exit":  lambda v: f"${v:,.4f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
            "P&L %": lambda v: f"{v:+.1f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
            "P&L $": lambda v: f"${v:+.2f}"  if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
            "Invested": lambda v: f"${v:.2f}"  if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
            "Conviction": lambda v: f"{v:.2f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
        }, na_rep="-")
    )


# ── tabs ──────────────────────────────────────────────────────────────────────

st.title("₿ Crypto Trading Bot Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshes every 60s")

tab_open, tab_resolved, tab_pnl, tab_lessons = st.tabs(
    ["Open Positions", "Closed Trades", "P&L Analytics", "Lessons"]
)

positions = load_open_positions()
resolved = load_resolved_trades()
lessons = load_lessons()
tracker = PnLTracker(RESOLVED_FILE)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_open:
    if not positions:
        st.info("No open positions found.")
    else:
        open_df = open_positions_df(positions)

        total_invested = open_df["Invested"].sum()
        total_unrlzd_usd = open_df["Unrlzd $"].sum() if "Unrlzd $" in open_df else 0
        total_unrlzd_pct = (total_unrlzd_usd / total_invested * 100) if total_invested > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open Positions", len(positions))
        c2.metric("Total Invested", f"${total_invested:.2f}")
        c3.metric("Unrealised P&L", f"${total_unrlzd_usd:+.2f}", f"{total_unrlzd_pct:+.1f}%")
        avg_conv = open_df["Conviction"].mean()
        c4.metric("Avg Conviction", f"{avg_conv:.2f}" if not pd.isna(avg_conv) else "-")

        st.divider()

        # Core table — always show
        fmt_safe = lambda v, fmt: fmt.format(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-"
        st.dataframe(
            open_df[["Coin", "Sector", "Direction", "Entry", "Now",
                      "Unrlzd %", "Unrlzd $", "Invested", "Conviction", "Horizon", "Open Since"]]
            .style
            .applymap(color_pnl, subset=["Unrlzd %", "Unrlzd $"])
            .format({
                "Entry":      lambda v: fmt_safe(v, "${:,.4f}"),
                "Now":        lambda v: fmt_safe(v, "${:,.4f}"),
                "Unrlzd %":   lambda v: fmt_safe(v, "{:+.1f}%"),
                "Unrlzd $":   lambda v: fmt_safe(v, "${:+.2f}"),
                "Invested":   lambda v: fmt_safe(v, "${:.2f}"),
                "Conviction": lambda v: fmt_safe(v, "{:.2f}"),
            }, na_rep="-"),
            use_container_width=True,
            height=400,
            column_config={"Coin": st.column_config.TextColumn(width="large")},
            hide_index=True,
        )

        # Volatility / trailing stop detail table
        vol_cols = ["Coin", "Vol/day", "Stop×", "Vol Risk", "Trailing Stop", "HWM"]
        vol_df = open_df[vol_cols].copy()
        has_vol_data = vol_df["Vol/day"].notna().any()
        has_trailing = vol_df["Trailing Stop"].notna().any()

        if has_vol_data or has_trailing:
            st.divider()
            st.subheader("Risk Detail")
            st.dataframe(
                vol_df.style.format({
                    "Vol/day":      lambda v: f"{v:.1f}%/d" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
                    "Stop×":        lambda v: f"{v:.1f}x"   if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
                    "Trailing Stop":lambda v: f"${v:,.4f}"  if v is not None and not (isinstance(v, float) and pd.isna(v)) else "not active",
                    "HWM":          lambda v: f"${v:,.4f}"  if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-",
                }, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )

        # Charts
        if len(open_df) > 1:
            st.divider()
            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.subheader("Exposure by Coin")
                fig = px.pie(
                    open_df, values="Invested", names="Coin", hole=0.4,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_bar:
                st.subheader("Unrealised P&L by Position")
                bar_df = open_df[["Coin", "Unrlzd $"]].copy()
                bar_df = bar_df.sort_values("Unrlzd $")
                colors = ["#00A86B" if v >= 0 else "#FF6B6B" for v in bar_df["Unrlzd $"]]
                fig2 = go.Figure(go.Bar(
                    x=bar_df["Unrlzd $"], y=bar_df["Coin"], orientation="h",
                    marker_color=colors,
                ))
                fig2.update_layout(
                    margin=dict(l=0, t=20, b=20), yaxis_title="",
                    xaxis_title="Unrealised P&L ($)",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Sector breakdown
        if "Sector" in open_df.columns and open_df["Sector"].notna().any():
            st.divider()
            st.subheader("Sector Exposure")
            sector_df = (
                open_df.groupby("Sector")
                .agg(Positions=("Invested", "count"), Invested=("Invested", "sum"))
                .reset_index()
                .sort_values("Positions", ascending=False)
            )
            col_sec_tbl, col_sec_pie = st.columns([1, 2])
            with col_sec_tbl:
                st.dataframe(
                    sector_df.style.format({"Invested": "${:.2f}"}),
                    hide_index=True, use_container_width=True,
                )
            with col_sec_pie:
                fig_sec = px.pie(sector_df, values="Positions", names="Sector", hole=0.4)
                fig_sec.update_traces(textposition="inside", textinfo="percent+label")
                fig_sec.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_sec, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLOSED TRADES
# ══════════════════════════════════════════════════════════════════════════════

with tab_resolved:
    if not resolved:
        st.info("No closed trades found.")
    else:
        res_df = resolved_trades_df(resolved)

        pnl_known = res_df.dropna(subset=["P&L $"])
        profitable = pnl_known[pnl_known["P&L $"] > 0]
        unprofitable = pnl_known[pnl_known["P&L $"] < 0]
        win_rate = len(profitable) / len(pnl_known) * 100 if len(pnl_known) > 0 else 0
        total_pnl = res_df["P&L $"].sum()
        total_invested_r = res_df["Invested"].sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", len(res_df))
        c2.metric("Wins / Losses", f"{len(profitable)} / {len(unprofitable)}")
        c3.metric("Win Rate", f"{win_rate:.0f}%")
        c4.metric("Total P&L", f"${total_pnl:+.2f}")
        roi = total_pnl / total_invested_r * 100 if total_invested_r > 0 else 0
        c5.metric("ROI", f"{roi:+.1f}%")

        st.divider()

        # Filters
        col_f1, col_f2 = st.columns(2)
        results_r = sorted(res_df["Result"].unique())
        sel_results_r = col_f1.multiselect("Result", results_r, default=results_r, key="res_result")

        dates = res_df["Date"].dropna()
        if len(dates) > 0:
            min_date = pd.to_datetime(dates.min()).date()
            max_date = pd.to_datetime(dates.max()).date()
            date_range = col_f2.date_input("Date range", value=(min_date, max_date), key="res_date")
        else:
            date_range = None

        mask = res_df["Result"].isin(sel_results_r)
        if date_range and len(date_range) == 2:
            mask &= (pd.to_datetime(res_df["Date"]) >= pd.Timestamp(date_range[0])) & \
                    (pd.to_datetime(res_df["Date"]) <= pd.Timestamp(date_range[1]))

        filtered_r = res_df[mask]

        st.dataframe(
            style_resolved_df(filtered_r),
            use_container_width=True,
            height=420,
            column_config={"Coin": st.column_config.TextColumn(width="medium")},
            hide_index=True,
        )

        # Best / worst
        st.divider()
        col_best, col_worst = st.columns(2)
        sorted_by_pnl = filtered_r.dropna(subset=["P&L $"]).sort_values("P&L $", ascending=False)
        with col_best:
            st.subheader("Best Trades")
            top5 = sorted_by_pnl.head(5)[["Date", "Coin", "P&L $", "P&L %"]]
            st.dataframe(
                top5.style.applymap(color_pnl, subset=["P&L $", "P&L %"])
                .format({"P&L $": "${:+.2f}", "P&L %": "{:+.1f}%"}, na_rep="-"),
                hide_index=True, use_container_width=True,
            )
        with col_worst:
            st.subheader("Worst Trades")
            bot5 = sorted_by_pnl.tail(5)[["Date", "Coin", "P&L $", "P&L %"]]
            st.dataframe(
                bot5.style.applymap(color_pnl, subset=["P&L $", "P&L %"])
                .format({"P&L $": "${:+.2f}", "P&L %": "{:+.1f}%"}, na_rep="-"),
                hide_index=True, use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — P&L ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_pnl:
    if not resolved:
        st.info("No closed trades to analyse yet.")
    else:
        all_time = tracker.all_time_summary()
        daily = tracker.daily_summary(days=90)
        monthly = tracker.monthly_summary()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("All-Time P&L", f"${all_time['total_pnl_usd']:+.2f}")
        c2.metric("Total Invested", f"${all_time['total_invested']:.2f}")
        roi_val = all_time['roi_pct']
        c3.metric("ROI", f"{roi_val:+.1f}%" if roi_val is not None else "-")
        wr = all_time['win_rate_pct']
        c4.metric("Win Rate", f"{wr:.0f}%" if wr is not None else "-")
        c5.metric("Total Trades", f"{all_time['total_trades']}  ({all_time['wins']}W / {all_time['losses']}L)")

        st.divider()

        # Cumulative P&L
        st.subheader("Cumulative P&L")
        res_df_pnl = resolved_trades_df(resolved).dropna(subset=["P&L $"]).copy()
        if not res_df_pnl.empty:
            res_df_pnl["Date"] = pd.to_datetime(res_df_pnl["Date"])
            res_df_pnl = res_df_pnl.sort_values("Date")
            res_df_pnl["Cumulative P&L"] = res_df_pnl["P&L $"].cumsum()
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=res_df_pnl["Date"], y=res_df_pnl["Cumulative P&L"],
                mode="lines+markers", fill="tozeroy",
                line=dict(color="#F7931A", width=2),
                fillcolor="rgba(247,147,26,0.15)",
                hovertemplate="<b>%{x|%b %d}</b><br>Cumulative P&L: $%{y:+.2f}<extra></extra>",
            ))
            fig_cum.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
            fig_cum.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="", yaxis_title="P&L ($)")
            st.plotly_chart(fig_cum, use_container_width=True)

        st.divider()

        # Daily / Monthly bars
        col_d, col_m = st.columns(2)

        with col_d:
            st.subheader("Daily P&L (last 30 days)")
            daily_df = pd.DataFrame([{"Date": k, **v} for k, v in daily.items()])
            if not daily_df.empty:
                daily_df["Date"] = pd.to_datetime(daily_df["Date"])
                daily_df = daily_df.sort_values("Date")
                daily_df["Color"] = daily_df["pnl_usd"].apply(lambda x: "#00A86B" if x >= 0 else "#FF6B6B")
                fig_d = go.Figure(go.Bar(
                    x=daily_df["Date"], y=daily_df["pnl_usd"], marker_color=daily_df["Color"],
                    hovertemplate="<b>%{x|%b %d}</b><br>P&L: $%{y:+.2f}<extra></extra>",
                ))
                fig_d.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
                fig_d.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="P&L ($)")
                st.plotly_chart(fig_d, use_container_width=True)

        with col_m:
            st.subheader("Monthly P&L")
            monthly_df = pd.DataFrame([{"Month": k, **v} for k, v in monthly.items()])
            if not monthly_df.empty:
                monthly_df = monthly_df.sort_values("Month")
                monthly_df["Color"] = monthly_df["pnl_usd"].apply(lambda x: "#00A86B" if x >= 0 else "#FF6B6B")
                fig_m = go.Figure(go.Bar(
                    x=monthly_df["Month"], y=monthly_df["pnl_usd"], marker_color=monthly_df["Color"],
                    hovertemplate="<b>%{x}</b><br>P&L: $%{y:+.2f}<extra></extra>",
                ))
                fig_m.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
                fig_m.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="P&L ($)")
                st.plotly_chart(fig_m, use_container_width=True)

        st.divider()

        # P&L vs Conviction scatter
        col_conv, col_dist = st.columns(2)

        with col_conv:
            st.subheader("P&L vs. Conviction")
            res_df_c = resolved_trades_df(resolved).dropna(subset=["Conviction", "P&L $"])
            if not res_df_c.empty:
                fig_conv = px.scatter(
                    res_df_c, x="Conviction", y="P&L $",
                    color="Result", color_discrete_map=RESULT_COLORS,
                    size="Invested",
                    hover_data=["Coin", "P&L %"],
                )
                fig_conv.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
                fig_conv.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig_conv, use_container_width=True)

        with col_dist:
            st.subheader("P&L Distribution")
            pnl_vals = res_df_pnl["P&L $"].dropna() if not res_df_pnl.empty else pd.Series(dtype=float)
            if not pnl_vals.empty:
                fig_hist = px.histogram(
                    pnl_vals, nbins=20, color_discrete_sequence=["#F7931A"],
                    labels={"value": "P&L ($)", "count": "# Trades"},
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="#888")
                fig_hist.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LESSONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_lessons:
    if not lessons:
        st.info("No lessons recorded yet. Lessons are generated after positions are closed.")
    else:
        for entry in reversed(lessons[-15:]):
            date = entry.get("date", "?")
            wr = entry.get("win_rate_pct")
            pnl = entry.get("pnl_usd")
            resolved_count = entry.get("resolved_count", 0)

            wr_str = f"Win rate: {wr}%" if wr is not None else "no data"
            pnl_str = f"  |  P&L: ${pnl:+.2f}" if pnl is not None else ""

            label = f"{date}  —  {resolved_count} closed  |  {wr_str}{pnl_str}"

            with st.expander(label, expanded=(entry == lessons[-1])):
                col_w, col_d = st.columns(2)

                with col_w:
                    what_worked = entry.get("what_worked", [])
                    if what_worked:
                        st.markdown("**What worked**")
                        for item in what_worked:
                            st.markdown(f"- {item}")

                with col_d:
                    what_didnt = entry.get("what_didnt_work", [])
                    if what_didnt:
                        st.markdown("**What didn't work**")
                        for item in what_didnt:
                            st.markdown(f"- {item}")

                lesson_list = entry.get("lessons", [])
                if lesson_list:
                    st.markdown("**Lessons for next session**")
                    for item in lesson_list:
                        st.markdown(f"- {item}")

                rq = entry.get("reasoning_quality")
                if rq:
                    st.caption(f"Reasoning quality: {rq}")
