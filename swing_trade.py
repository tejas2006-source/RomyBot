"""
swing_trade.py — Rule-based swing-trade analyzer for RomyBot.

Implements a discretionary swing-trading checklist as a deterministic screener.
Given a ticker, it pulls daily bars and reports trade eligibility, a suggested
entry price, a non-negotiable stop-loss, and the exit rule.

Rules encoded (from the strategy spec):
  1. Base / consolidation : require a "big daily base" — at least ~2 weeks
     (10 trading days) of tight consolidation before considering a trade.
  2. Key levels / stop    : entry near the recent breakout level; the
     non-negotiable stop-loss is the LOW OF THE (entry) DAY.
  3. Moving averages       : use 8 & 21 EMA for trend/momentum; the 8-day MA
     is the final-exit signal (exit when price closes below the 8-day MA).
  4. Volume                : require accumulation (up-day volume > down-day
     volume) AND today's volume >= 1.5x (50% over) the 10-day average volume
     as entry confirmation.

Data source: yfinance (no API keys needed; matches the existing repo stack).

Usage:
    python swing_trade.py AAPL
    python swing_trade.py            # prompts for a ticker
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
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


# ------------------------------- data model ------------------------------ #

@dataclass
class Analysis:
    ticker: str
    price: float
    day_low: float
    day_high: float
    ema8: float
    ema21: float
    avg_vol_10: float
    today_vol: float
    accumulation: bool
    base_ok: bool
    base_range_pct: float
    base_high: float
    trend_up: bool
    vol_ok: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.base_ok and self.trend_up and self.vol_ok and self.accumulation


# ------------------------------- indicators ------------------------------ #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


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

    if data is None or data.empty or len(data) < EMA_SLOW + 1:
        raise ValueError(
            f"Not enough daily data for {ticker!r} to run the analysis."
        )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    ema8 = ema(close, EMA_FAST)
    ema21 = ema(close, EMA_SLOW)

    price = float(close.iloc[-1])
    day_low = float(low.iloc[-1])
    day_high = float(high.iloc[-1])
    last_ema8 = float(ema8.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])

    # --- Volume: 10-day avg (excluding today) and 50%-over confirmation ---
    avg_vol_10 = float(volume.iloc[-(VOL_LOOKBACK + 1):-1].mean())
    today_vol = float(volume.iloc[-1])
    vol_ok = today_vol >= VOL_MULTIPLIER * avg_vol_10

    # --- Accumulation: up-day volume vs down-day volume over lookback ---
    window = data.iloc[-(VOL_LOOKBACK + 1):]
    daily_change = window["Close"].diff()
    up_vol = window["Volume"][daily_change > 0].sum()
    down_vol = window["Volume"][daily_change < 0].sum()
    accumulation = bool(up_vol > down_vol)

    # --- Base / consolidation: tight range over the prior ~2 weeks ---
    base = data.iloc[-(CONSOLIDATION_DAYS + 1):-1]   # prior window, not today
    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())
    base_range_pct = (base_high - base_low) / base_low if base_low else float("inf")
    base_ok = base_range_pct <= CONSOLIDATION_MAX_RANGE

    # --- Trend / momentum: 8 EMA above 21 EMA and price above 8 EMA ---
    trend_up = (last_ema8 > last_ema21) and (price > last_ema8)

    a = Analysis(
        ticker=ticker,
        price=price,
        day_low=day_low,
        day_high=day_high,
        ema8=last_ema8,
        ema21=last_ema21,
        avg_vol_10=avg_vol_10,
        today_vol=today_vol,
        accumulation=accumulation,
        base_ok=base_ok,
        base_range_pct=base_range_pct,
        base_high=base_high,
        trend_up=trend_up,
        vol_ok=vol_ok,
    )

    if not base_ok:
        a.reasons.append(
            f"No tight base: {CONSOLIDATION_DAYS}-day range is "
            f"{base_range_pct*100:.1f}% (need <= {CONSOLIDATION_MAX_RANGE*100:.0f}%)."
        )
    if not trend_up:
        a.reasons.append(
            "Trend not up: need 8EMA > 21EMA and price above the 8EMA."
        )
    if not vol_ok:
        a.reasons.append(
            f"Volume weak: today {today_vol:,.0f} vs need "
            f">= {VOL_MULTIPLIER:.1f}x 10d avg ({VOL_MULTIPLIER*avg_vol_10:,.0f})."
        )
    if not accumulation:
        a.reasons.append("No accumulation: down-day volume >= up-day volume.")

    return a


# ------------------------------- reporting ------------------------------- #

def report(a: Analysis) -> str:
    lines = []
    lines.append(f"=== Swing-Trade Analysis: {a.ticker} ===")
    lines.append(f"Price:            {a.price:.2f}")
    lines.append(f"Day range:        {a.day_low:.2f} - {a.day_high:.2f}")
    lines.append(f"8 EMA / 21 EMA:   {a.ema8:.2f} / {a.ema21:.2f}")
    lines.append(f"10d avg volume:   {a.avg_vol_10:,.0f}")
    lines.append(f"Today's volume:   {a.today_vol:,.0f} "
                 f"({a.today_vol / a.avg_vol_10:.2f}x avg)" if a.avg_vol_10 else "")
    lines.append("")
    lines.append("Checklist:")
    lines.append(f"  [{'x' if a.base_ok else ' '}] Tight base "
                 f"({CONSOLIDATION_DAYS}d range {a.base_range_pct*100:.1f}%)")
    lines.append(f"  [{'x' if a.trend_up else ' '}] Uptrend (8EMA > 21EMA, price > 8EMA)")
    lines.append(f"  [{'x' if a.vol_ok else ' '}] Volume >= "
                 f"{VOL_MULTIPLIER:.0%} over 10d avg")
    lines.append(f"  [{'x' if a.accumulation else ' '}] Accumulation "
                 f"(up-vol > down-vol)")
    lines.append("")

    if a.eligible:
        # Entry: breakout over the base high (use max of base high / current price).
        entry = max(a.base_high, a.price)
        stop = a.day_low  # non-negotiable: low of the (entry) day
        risk = entry - stop
        lines.append(">>> SETUP VALID — swing trade qualifies.")
        lines.append(f"  ENTRY (buy-stop): {entry:.2f}  (break of base high {a.base_high:.2f})")
        lines.append(f"  STOP-LOSS:        {stop:.2f}  (low of the day — non-negotiable)")
        if risk > 0:
            lines.append(f"  Risk/share:       {risk:.2f}  "
                         f"({risk / entry * 100:.1f}% to stop)")
            lines.append(f"  R-target (2R):    {entry + 2 * risk:.2f}")
        lines.append(f"  EXIT signal:      close below the 8-day MA "
                     f"(currently {a.ema8:.2f}).")
    else:
        lines.append(">>> NO TRADE — setup does not qualify.")
        for r in a.reasons:
            lines.append(f"  - {r}")
        lines.append("")
        lines.append("If it later qualifies, the plan would be:")
        lines.append(f"  ENTRY:     break of base high (~{a.base_high:.2f})")
        lines.append("  STOP:      low of the entry day (non-negotiable)")
        lines.append("  EXIT:      close below the 8-day MA")

    return "\n".join(line for line in lines if line is not None)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        ticker = argv[1]
    else:
        ticker = input("Enter the stock ticker: ")

    try:
        a = analyze(ticker)
    except Exception as exc:  # noqa: BLE001
        print(f"Error analyzing {ticker!r}: {exc}", file=sys.stderr)
        return 1

    print(report(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
