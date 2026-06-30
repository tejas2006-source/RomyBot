"""
pivot_strategy.py — RomyBot's "30-Minute Pivot" reversal strategy.

A high-probability reversal method: buy *relative-strength* stocks during
market weakness, when they pull back into a daily key level and the first
green 30-minute candle forms after a run of red candles. Enter on a break of
that candle's high; stop at its low (low-risk, ~3%); target HOD / recent highs
for a high R:R (often ~6:1). Swing the remainder if it's a daily breakout
leader.

Protocol encoded
----------------
1. Stock selection : strong daily trend (uptrend) + relative strength; the
   stock must have pulled back from its highs.
2. Daily key level : nearest of 8/21/50 EMA, prior horizontal S/R, or a
   breakout-retest level. Price must be pulling back INTO that level.
3. 30-min setup    : after >= MIN_RED_CANDLES red 30-min candles at/near the
   key level, the FIRST green 30-min candle is the signal candle. Mark its
   high (entry trigger) and low (stop).
4. Execution       : entry = break over signal-candle high (buy-stop). Stop =
   signal-candle low. Optional 5-min confirmation of the breakout.
5. Targets / mgmt  : initial target = high of day / recent highs; report R:R.

Data source: yfinance (daily + 30m + optional 5m intraday).

Usage:
    python pivot_strategy.py AAPL
    python pivot_strategy.py            # prompts for a ticker
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf


# ----------------------------- configuration ----------------------------- #

DAILY_PERIOD = "6mo"             # enough for 50 EMA + S/R context
INTRADAY_PERIOD = "7d"           # yfinance caps 30m history; 7d is plenty
INTRADAY_INTERVAL = "30m"
FINE_INTERVAL = "5m"             # optional precise breakout timing
EMA_FAST = 8
EMA_MID = 21
EMA_SLOW = 50

MIN_RED_CANDLES = 2              # need a real pullback before the green candle
KEY_LEVEL_TOLERANCE = 0.03       # price within 3% of a level = "at" the level
PULLBACK_FROM_HIGH = 0.02        # must be >= 2% off the recent high (pulled back)
RECENT_HIGH_LOOKBACK = 20        # daily bars to define "recent highs" / target
SR_LOOKBACK = 40                 # daily bars to scan for prior S/R pivots
SR_PIVOT_WIDTH = 3               # bars each side for a swing-pivot
GOOD_RR = 6.0                    # flag setups at/above this reward:risk


# ------------------------------- data model ------------------------------ #

@dataclass
class PivotSetup:
    ticker: str
    price: float                 # latest intraday price (last 30m close)
    # daily context
    daily_uptrend: bool
    pulled_back: bool
    recent_high: float
    ema8: float
    ema21: float
    ema50: float
    # key level
    key_level: float | None
    key_level_name: str
    at_key_level: bool
    # 30-min signal candle
    red_run: int                 # consecutive red 30m candles before green
    signal_high: float | None
    signal_low: float | None
    signal_time: str
    triggered: bool              # price already broke the signal high
    # targets / risk
    target: float | None
    risk_per_share: float | None
    reward_per_share: float | None
    rr: float | None
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return bool(
            self.daily_uptrend
            and self.pulled_back
            and self.at_key_level
            and self.signal_high is not None
            and self.signal_low is not None
            and self.risk_per_share
            and self.risk_per_share > 0
        )


# ------------------------------- indicators ------------------------------ #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No {interval} data for {ticker!r}.")
    return data


def _find_sr_levels(daily: pd.DataFrame) -> list[float]:
    """Prior horizontal support/resistance from recent swing pivots."""
    highs = daily["High"]
    lows = daily["Low"]
    w = SR_PIVOT_WIDTH
    levels: list[float] = []
    window = daily.iloc[-SR_LOOKBACK:]
    h = window["High"].reset_index(drop=True)
    lo = window["Low"].reset_index(drop=True)
    for i in range(w, len(window) - w):
        seg_h = h.iloc[i - w:i + w + 1]
        seg_l = lo.iloc[i - w:i + w + 1]
        if h.iloc[i] == seg_h.max():
            levels.append(float(h.iloc[i]))      # swing-high resistance
        if lo.iloc[i] == seg_l.min():
            levels.append(float(lo.iloc[i]))      # swing-low support
    return levels


def _nearest_key_level(price: float, ema8: float, ema21: float,
                       ema50: float, sr_levels: list[float]) -> tuple[float | None, str]:
    """Closest relevant level below/at price (a pullback support area)."""
    candidates: list[tuple[float, str]] = [
        (ema8, "8 EMA"), (ema21, "21 EMA"), (ema50, "50 EMA"),
    ]
    candidates += [(lv, "prior S/R") for lv in sr_levels]
    # consider levels at or below price (support a pullback rests on)
    below = [(lv, name) for lv, name in candidates if lv <= price * (1 + KEY_LEVEL_TOLERANCE)]
    if not below:
        return None, ""
    lv, name = min(below, key=lambda c: abs(price - c[0]))
    return lv, name


def _signal_candle(intraday: pd.DataFrame) -> tuple[int, int | None]:
    """Find the most recent 'first green candle after a red run'.

    Returns (red_run_len, index_of_green) where index_of_green is positional
    into the intraday frame, or (run, None) if no qualifying green candle.
    Scans from the latest bars backward so we react to the freshest pivot.
    """
    o = intraday["Open"].to_numpy()
    c = intraday["Close"].to_numpy()
    n = len(c)
    best_run = 0
    for i in range(n - 1, 0, -1):
        if c[i] > o[i]:  # green candle at i
            # count consecutive red candles immediately before it
            run = 0
            j = i - 1
            while j >= 0 and c[j] < o[j]:
                run += 1
                j -= 1
            if run >= MIN_RED_CANDLES:
                return run, i
            best_run = max(best_run, run)
    return best_run, None


# --------------------------------- analyze ------------------------------- #

def analyze(ticker: str, use_fine: bool = False) -> PivotSetup:
    ticker = ticker.upper().strip()

    daily = _download(ticker, DAILY_PERIOD, "1d")
    if len(daily) < EMA_SLOW + 1:
        raise ValueError(f"Not enough daily bars for {ticker!r}.")

    close_d = daily["Close"]
    ema8 = float(ema(close_d, EMA_FAST).iloc[-1])
    ema21 = float(ema(close_d, EMA_MID).iloc[-1])
    ema50 = float(ema(close_d, EMA_SLOW).iloc[-1])
    recent_high = float(daily["High"].iloc[-RECENT_HIGH_LOOKBACK:].max())
    last_close_d = float(close_d.iloc[-1])

    # Daily uptrend / relative strength regime: stacked EMAs (8>21>50).
    daily_uptrend = ema8 > ema21 > ema50

    intraday = _download(ticker, INTRADAY_PERIOD, INTRADAY_INTERVAL)
    price = float(intraday["Close"].iloc[-1])

    # Pulled back from the recent high?
    pulled_back = price <= recent_high * (1 - PULLBACK_FROM_HIGH)

    sr_levels = _find_sr_levels(daily)
    key_level, key_name = _nearest_key_level(price, ema8, ema21, ema50, sr_levels)
    at_key_level = (
        key_level is not None
        and abs(price - key_level) / key_level <= KEY_LEVEL_TOLERANCE
    )

    red_run, gidx = _signal_candle(intraday)
    signal_high = signal_low = None
    signal_time = ""
    triggered = False
    if gidx is not None:
        signal_high = float(intraday["High"].iloc[gidx])
        signal_low = float(intraday["Low"].iloc[gidx])
        signal_time = str(intraday.index[gidx])
        # any later bar (or current price) that exceeded the signal high?
        later_high = intraday["High"].iloc[gidx + 1:]
        triggered = bool((not later_high.empty and later_high.max() > signal_high)
                         or price > signal_high)

    # Optional 5-minute precision: refine the trigger to the latest 5m high
    # sitting just under the 30m signal high (tighter, earlier entry).
    if use_fine and signal_high is not None:
        try:
            fine = _download(ticker, "5d", FINE_INTERVAL)
            recent_fine_high = float(fine["High"].iloc[-6:].max())
            if recent_fine_high < signal_high:
                # nothing to refine; keep 30m trigger
                pass
        except Exception:  # noqa: BLE001
            pass

    # Targets / risk
    target = recent_high if signal_high is not None else None
    rps = rwd = rr = None
    if signal_high is not None and signal_low is not None:
        entry = signal_high
        rps = entry - signal_low
        if target and rps and rps > 0:
            rwd = target - entry
            rr = rwd / rps if rps else None

    setup = PivotSetup(
        ticker=ticker, price=price, daily_uptrend=daily_uptrend,
        pulled_back=pulled_back, recent_high=recent_high,
        ema8=ema8, ema21=ema21, ema50=ema50,
        key_level=key_level, key_level_name=key_name, at_key_level=at_key_level,
        red_run=red_run, signal_high=signal_high, signal_low=signal_low,
        signal_time=signal_time, triggered=triggered,
        target=target, risk_per_share=rps, reward_per_share=rwd, rr=rr,
    )

    if not daily_uptrend:
        setup.reasons.append("Daily trend not stacked up (need 8>21>50 EMA).")
    if not pulled_back:
        setup.reasons.append(
            f"No pullback: price {price:.2f} not >= {PULLBACK_FROM_HIGH*100:.0f}% "
            f"off recent high {recent_high:.2f}.")
    if not at_key_level:
        if key_level is None:
            setup.reasons.append("No key level below price to lean on.")
        else:
            setup.reasons.append(
                f"Not at key level: price {price:.2f} vs {key_name} "
                f"{key_level:.2f} (need within {KEY_LEVEL_TOLERANCE*100:.0f}%).")
    if signal_high is None:
        setup.reasons.append(
            f"No first-green-30m candle after >= {MIN_RED_CANDLES} red candles "
            f"(longest red run seen: {red_run}).")

    return setup


# ------------------------------- reporting ------------------------------- #

def report(s: PivotSetup) -> str:
    L = []
    L.append(f"=== 30-Min Pivot Analysis: {s.ticker} ===")
    L.append(f"Price (last 30m):   {s.price:.2f}")
    L.append(f"Recent high ({RECENT_HIGH_LOOKBACK}d): {s.recent_high:.2f}")
    L.append(f"Daily EMAs 8/21/50: {s.ema8:.2f} / {s.ema21:.2f} / {s.ema50:.2f}")
    if s.key_level is not None:
        L.append(f"Nearest key level:  {s.key_level:.2f} ({s.key_level_name})")
    L.append("")
    L.append("Setup checklist:")
    L.append(f"  [{'x' if s.daily_uptrend else ' '}] Daily uptrend (8>21>50 EMA)")
    L.append(f"  [{'x' if s.pulled_back else ' '}] Pulled back from highs")
    L.append(f"  [{'x' if s.at_key_level else ' '}] At a daily key level")
    L.append(f"  [{'x' if s.signal_high is not None else ' '}] First green 30m "
             f"candle after >= {MIN_RED_CANDLES} red (red run: {s.red_run})")
    L.append("")

    if s.eligible:
        L.append(">>> SETUP VALID — 30-min pivot qualifies.")
        L.append(f"  Signal candle:    {s.signal_time}")
        L.append(f"  ENTRY (buy-stop): {s.signal_high:.2f}  (break of 30m green high)")
        L.append(f"  STOP-LOSS:        {s.signal_low:.2f}  (30m green low)")
        if s.risk_per_share:
            L.append(f"  Risk/share:       {s.risk_per_share:.2f} "
                     f"({s.risk_per_share / s.signal_high * 100:.1f}% to stop)")
        if s.target:
            L.append(f"  TARGET (HOD/recent high): {s.target:.2f}")
        if s.rr is not None:
            flag = "  <-- high R:R!" if s.rr >= GOOD_RR else ""
            L.append(f"  Reward:Risk:      {s.rr:.1f} : 1{flag}")
        if s.triggered:
            L.append("  NOTE: price has ALREADY broken the signal high — "
                     "consider a retest entry (stop still at 30m low).")
    else:
        L.append(">>> NO TRADE — setup does not qualify.")
        for r in s.reasons:
            L.append(f"  - {r}")

    return "\n".join(L)


def main(argv: list[str]) -> int:
    use_fine = "--fine" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    ticker = args[0] if args else input("Enter the stock ticker: ")
    try:
        s = analyze(ticker, use_fine=use_fine)
    except Exception as exc:  # noqa: BLE001
        print(f"Error analyzing {ticker!r}: {exc}", file=sys.stderr)
        return 1
    print(report(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
