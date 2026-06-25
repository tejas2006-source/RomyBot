"""
swing_trade.py — Rule-based swing-trade analyzer for RomyBot (v2).

Rewrite notes (2026-06-25):
  * FIXED the volume bug. v1 compared the LAST bar's volume to a full-day
    10-day average. When run intraday, yfinance returns an in-progress
    partial bar, so "today's volume" was a fraction of a session and the
    1.5x test failed on literally everything. v2 evaluates only COMPLETED
    daily bars (drops a partial last bar), so volume is apples-to-apples.
  * DECOUPLED base vs trend. v1 required price already extended above the
    8 EMA *and* a tight base — those fight each other. v2's trend check is
    just 8 EMA > 21 EMA (momentum regime). The breakout itself is the entry
    trigger, not a precondition.
  * SCORED checklist instead of a 4-way AND wall. Each signal contributes a
    point; a setup is eligible at >= PASS_THRESHOLD (default 3 of 4). The
    base breakout structure (base_ok) is still required as a gate, since the
    entry/stop are defined off it.

Rules encoded:
  1. Base / consolidation : >= ~2 weeks (10 trading days) of tight range.
  2. Key levels / stop    : entry = break of base high; stop = low of the
     last completed day (non-negotiable).
  3. Moving averages      : 8 & 21 EMA for trend; exit = close below 8 EMA.
  4. Volume               : last completed day's volume >= 1.5x the prior
     10-day average (confirmation, scored — not a hard block).
  5. Accumulation         : up-day volume > down-day volume (scored).

Data source: yfinance.

Usage:
    python swing_trade.py AAPL
    python swing_trade.py            # prompts for a ticker
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ----------------------------- configuration ----------------------------- #

CONSOLIDATION_DAYS = 10          # ~2 weeks of trading days = "base"
CONSOLIDATION_MAX_RANGE = 0.15   # base is "tight" if range <= 15% of its low
VOL_LOOKBACK = 10                # 10-day average volume window
VOL_MULTIPLIER = 1.5             # require 50% over the 10-day average volume
EMA_FAST = 8
EMA_SLOW = 21
HISTORY_PERIOD = "6mo"           # enough bars for 21 EMA + a 10-day base
PASS_THRESHOLD = 3               # need >= this many of 4 scored signals
PARTIAL_BAR_MAX_AGE_H = 20       # if last bar's date is "today" & session not
                                 # closed, treat it as partial and drop it


# ------------------------------- data model ------------------------------ #

@dataclass
class Analysis:
    ticker: str
    price: float                 # last COMPLETED close
    day_low: float               # last completed day low (stop basis)
    day_high: float
    ema8: float
    ema21: float
    avg_vol_10: float
    today_vol: float             # last completed day's volume
    accumulation: bool
    base_ok: bool
    base_range_pct: float
    base_high: float
    trend_up: bool
    vol_ok: bool
    score: int = 0
    used_partial_drop: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        # base structure is a hard gate (entry/stop are defined off it);
        # the rest is a scored checklist.
        return self.base_ok and self.score >= PASS_THRESHOLD


# ------------------------------- indicators ------------------------------ #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _drop_partial_bar(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Drop a likely in-progress (partial) last daily bar.

    yfinance, when polled intraday, appends a partial bar dated 'today'.
    Its volume is a fraction of a full session and corrupts volume tests.
    We drop the last row when its date equals the current UTC date, on the
    assumption the US session for that date isn't fully booked yet.
    """
    if data.empty:
        return data, False
    last_ts = data.index[-1]
    last_date = pd.Timestamp(last_ts).date()
    today_utc = datetime.now(timezone.utc).date()
    if last_date >= today_utc:
        return data.iloc[:-1], True
    return data, False


def analyze(ticker: str) -> Analysis:
    ticker = ticker.upper().strip()
    data = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No daily data for {ticker!r}.")

    data, used_partial_drop = _drop_partial_bar(data)

    if len(data) < EMA_SLOW + 1:
        raise ValueError(
            f"Not enough completed daily bars for {ticker!r} to analyze."
        )

    close = data["Close"]
    low = data["Low"]
    high = data["High"]
    volume = data["Volume"]

    ema8 = ema(close, EMA_FAST)
    ema21 = ema(close, EMA_SLOW)

    price = float(close.iloc[-1])          # last COMPLETED close
    day_low = float(low.iloc[-1])
    day_high = float(high.iloc[-1])
    last_ema8 = float(ema8.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])

    # --- Volume: last completed bar vs prior 10 completed bars ---
    avg_vol_10 = float(volume.iloc[-(VOL_LOOKBACK + 1):-1].mean())
    today_vol = float(volume.iloc[-1])
    vol_ok = avg_vol_10 > 0 and today_vol >= VOL_MULTIPLIER * avg_vol_10

    # --- Accumulation: up-day vs down-day volume over lookback ---
    window = data.iloc[-(VOL_LOOKBACK + 1):]
    daily_change = window["Close"].diff()
    up_vol = window["Volume"][daily_change > 0].sum()
    down_vol = window["Volume"][daily_change < 0].sum()
    accumulation = bool(up_vol > down_vol)

    # --- Base / consolidation: tight range over the prior ~2 weeks ---
    base = data.iloc[-(CONSOLIDATION_DAYS + 1):-1]   # prior window, not last bar
    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())
    base_range_pct = (base_high - base_low) / base_low if base_low else float("inf")
    base_ok = base_range_pct <= CONSOLIDATION_MAX_RANGE

    # --- Trend regime: 8 EMA above 21 EMA (momentum up). Decoupled from
    #     "price already above 8 EMA" so a base breakout can still qualify. ---
    trend_up = last_ema8 > last_ema21

    # --- Scored checklist (4 signals) ---
    score = sum([bool(base_ok), bool(trend_up), bool(vol_ok), bool(accumulation)])

    a = Analysis(
        ticker=ticker, price=price, day_low=day_low, day_high=day_high,
        ema8=last_ema8, ema21=last_ema21, avg_vol_10=avg_vol_10,
        today_vol=today_vol, accumulation=accumulation, base_ok=base_ok,
        base_range_pct=base_range_pct, base_high=base_high, trend_up=trend_up,
        vol_ok=vol_ok, score=score, used_partial_drop=used_partial_drop,
    )

    if not base_ok:
        a.reasons.append(
            f"No tight base: {CONSOLIDATION_DAYS}-day range is "
            f"{base_range_pct*100:.1f}% (need <= {CONSOLIDATION_MAX_RANGE*100:.0f}%)."
        )
    if not trend_up:
        a.reasons.append("Trend regime down: 8EMA <= 21EMA.")
    if not vol_ok:
        a.reasons.append(
            f"Volume weak: last completed day {today_vol:,.0f} vs need "
            f">= {VOL_MULTIPLIER:.1f}x 10d avg ({VOL_MULTIPLIER*avg_vol_10:,.0f})."
        )
    if not accumulation:
        a.reasons.append("No accumulation: down-day volume >= up-day volume.")

    return a


# ------------------------------- reporting ------------------------------- #

def report(a: Analysis) -> str:
    lines = []
    lines.append(f"=== Swing-Trade Analysis: {a.ticker} ===")
    lines.append(f"Price (last close): {a.price:.2f}")
    lines.append(f"Day range:          {a.day_low:.2f} - {a.day_high:.2f}")
    lines.append(f"8 EMA / 21 EMA:     {a.ema8:.2f} / {a.ema21:.2f}")
    lines.append(f"10d avg volume:     {a.avg_vol_10:,.0f}")
    if a.avg_vol_10:
        lines.append(f"Last-day volume:    {a.today_vol:,.0f} "
                     f"({a.today_vol / a.avg_vol_10:.2f}x avg)")
    if a.used_partial_drop:
        lines.append("(note: dropped a partial in-progress bar)")
    lines.append("")
    lines.append(f"Checklist (score {a.score}/4, need {PASS_THRESHOLD}):")
    lines.append(f"  [{'x' if a.base_ok else ' '}] Tight base "
                 f"({CONSOLIDATION_DAYS}d range {a.base_range_pct*100:.1f}%) [GATE]")
    lines.append(f"  [{'x' if a.trend_up else ' '}] Trend regime (8EMA > 21EMA)")
    lines.append(f"  [{'x' if a.vol_ok else ' '}] Volume >= "
                 f"{VOL_MULTIPLIER:.0%} over 10d avg")
    lines.append(f"  [{'x' if a.accumulation else ' '}] Accumulation "
                 f"(up-vol > down-vol)")
    lines.append("")

    if a.eligible:
        entry = max(a.base_high, a.price)
        stop = a.day_low
        risk = entry - stop
        lines.append(">>> SETUP VALID — swing trade qualifies.")
        lines.append(f"  ENTRY (buy-stop): {entry:.2f}  (break of base high {a.base_high:.2f})")
        lines.append(f"  STOP-LOSS:        {stop:.2f}  (last-day low — non-negotiable)")
        if risk > 0:
            lines.append(f"  Risk/share:       {risk:.2f}  "
                         f"({risk / entry * 100:.1f}% to stop)")
            lines.append(f"  R-target (2R):    {entry + 2 * risk:.2f}")
        lines.append(f"  EXIT signal:      close below the 8-day MA "
                     f"(currently {a.ema8:.2f}).")
    else:
        lines.append(">>> NO TRADE — setup does not qualify.")
        if not a.base_ok:
            lines.append("  - base gate failed (entry/stop undefined).")
        for r in a.reasons:
            lines.append(f"  - {r}")

    return "\n".join(line for line in lines if line is not None)


def main(argv: list[str]) -> int:
    ticker = argv[1] if len(argv) > 1 else input("Enter the stock ticker: ")
    try:
        a = analyze(ticker)
    except Exception as exc:  # noqa: BLE001
        print(f"Error analyzing {ticker!r}: {exc}", file=sys.stderr)
        return 1
    print(report(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
