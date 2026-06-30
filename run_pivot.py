"""
run_pivot.py — Screen a watchlist with the 30-Minute Pivot strategy and place
buy-stop bracket orders on Alpaca PAPER for any setup that qualifies.

For each VALID pivot setup:
  - entry  : break of the first green 30m candle's HIGH (buy-stop trigger)
  - stop   : that candle's LOW (low-risk, ~3%)
  - target : high of day / recent highs (bracket take-profit)
  - qty    : risk-based size = floor(RISK_DOLLARS / (entry - stop)),
             capped at MAX_POSITION_FRACTION of equity.

Every evaluated ticker (placed or skipped) is logged to trades_log.csv.

Credentials come from the macOS keychain at runtime (never stored in code):
  alpaca-paper-key    / account alpaca-paper
  alpaca-paper-secret / account alpaca-paper

Usage:
  python run_pivot.py                 # default universe
  python run_pivot.py NVDA AAPL AMD   # custom tickers
  python run_pivot.py --dry-run       # screen + log only, place NO orders
  python run_pivot.py --fine          # use 5m for precise breakout timing
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

import requests

from pivot_strategy import analyze, PivotSetup


# ------------------------------ configuration ---------------------------- #

DEFAULT_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "AMD", "TSLA",
    "AMZN", "GOOGL", "META", "AVGO", "NFLX",
    "PLTR", "COIN", "SHOP", "UBER", "CRWD",
    "SMCI", "MU", "QCOM", "NOW", "PANW",
    "SNOW", "DDOG", "NET", "ABNB", "MRVL",
]

ALPACA_BASE = "https://paper-api.alpaca.markets"
RISK_FRACTION = 0.01           # risk 1% of account equity per trade
MAX_POSITION_FRACTION = 0.10   # cap any single position at 10% of equity
LOG_PATH = os.path.join(os.path.dirname(__file__), "trades_log.csv")

LOG_FIELDS = [
    "timestamp", "strategy", "ticker", "decision", "market_open", "price",
    "key_level", "key_level_name", "signal_time", "entry", "stop",
    "take_profit", "rr", "qty", "risk_per_share", "risk_dollars",
    "order_id", "status", "notes",
]
STRATEGY = "30min_pivot"


# ------------------------------- credentials ----------------------------- #

def _keychain(account: str, service: str) -> str:
    out = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True, text=True,
    )
    val = out.stdout.strip()
    if not val:
        raise RuntimeError(f"Missing keychain item: account={account} service={service}")
    return val


def alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": _keychain("alpaca-paper", "alpaca-paper-key"),
        "APCA-API-SECRET-KEY": _keychain("alpaca-paper", "alpaca-paper-secret"),
    }


# --------------------------------- Alpaca -------------------------------- #

def get_equity(headers: dict) -> float:
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=headers, timeout=15)
    r.raise_for_status()
    return float(r.json()["equity"])


def market_is_open(headers: dict) -> bool:
    r = requests.get(f"{ALPACA_BASE}/v2/clock", headers=headers, timeout=15)
    r.raise_for_status()
    return bool(r.json().get("is_open"))


def submit_bracket(headers: dict, ticker: str, qty: int,
                   entry: float, stop: float, take_profit: float) -> dict:
    """Buy-stop entry with attached stop-loss + take-profit (bracket)."""
    payload = {
        "symbol": ticker,
        "qty": str(qty),
        "side": "buy",
        "type": "stop",
        "stop_price": round(entry, 2),
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": round(take_profit, 2)},
        "stop_loss": {"stop_price": round(stop, 2)},
    }
    r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=headers,
                      json=payload, timeout=20)
    if r.status_code >= 300:
        return {"error": f"{r.status_code}: {r.text}"}
    return r.json()


# ------------------------------- trade logic ----------------------------- #

def size_position(equity: float, entry: float, stop: float) -> tuple[int, float, float]:
    risk_dollars = equity * RISK_FRACTION
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return 0, risk_per_share, risk_dollars
    qty = math.floor(risk_dollars / risk_per_share)
    max_notional = equity * MAX_POSITION_FRACTION
    qty_by_notional = math.floor(max_notional / entry)
    return min(qty, qty_by_notional), risk_per_share, risk_dollars


def log_row(row: dict) -> None:
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def process(ticker: str, headers: dict, equity: float,
            dry_run: bool, use_fine: bool, market_open: bool) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"timestamp": now, "strategy": STRATEGY, "ticker": ticker,
            "market_open": "open" if market_open else "closed"}

    try:
        s: PivotSetup = analyze(ticker, use_fine=use_fine)
    except Exception as exc:  # noqa: BLE001
        # Data/analysis error: report it, but do NOT log failures to the table.
        return {**base, "decision": "ERROR", "notes": str(exc)}

    # Shared strategy context logged for EVERY ticker, qualified or not.
    ctx = {
        **base, "price": f"{s.price:.2f}",
        "key_level": f"{s.key_level:.2f}" if s.key_level is not None else "",
        "key_level_name": s.key_level_name,
        "signal_time": s.signal_time,
        "rr": f"{s.rr:.1f}" if s.rr is not None else "",
    }

    if not s.eligible:
        row = {
            **ctx, "decision": "SKIP",
            "notes": "; ".join(s.reasons) or "did not qualify",
        }
        log_row(row)
        return row

    entry = float(s.signal_high)
    stop = float(s.signal_low)
    take_profit = float(s.target)
    qty, rps, risk_dollars = size_position(equity, entry, stop)

    row = {
        **ctx, "decision": "VALID",
        "entry": f"{entry:.2f}", "stop": f"{stop:.2f}",
        "take_profit": f"{take_profit:.2f}", "qty": qty,
        "risk_per_share": f"{rps:.2f}", "risk_dollars": f"{risk_dollars:.0f}",
        "notes": f"{s.key_level_name} level; "
                 f"{'TRIGGERED-retest' if s.triggered else 'pre-break'}",
    }

    if take_profit <= entry:
        row["decision"] = "SKIP"
        row["notes"] = "target at/below entry (no room) — skipped"
        log_row(row)
        return row

    if qty < 1:
        row["decision"] = "SKIP"
        row["notes"] = "position size < 1 share at risk budget"
        log_row(row)
        return row

    if dry_run:
        row["status"] = "DRY_RUN"
        row["notes"] = (row.get("notes", "") + " | dry-run: order not submitted").strip(" |")
        log_row(row)
        return row

    resp = submit_bracket(headers, ticker, qty, entry, stop, take_profit)
    if "error" in resp:
        # Order rejection: report it, but do NOT log failures to the table.
        row["decision"] = "ORDER_FAILED"
        row["status"] = "error"
        row["notes"] = resp["error"]
        return row
    row["order_id"] = resp.get("id", "")
    row["status"] = resp.get("status", "")
    row["notes"] = (row.get("notes", "") + " | bracket buy-stop submitted").strip(" |")
    log_row(row)
    return row


# ----------------------------------- main -------------------------------- #

def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    use_fine = "--fine" in argv
    tickers = [a for a in argv[1:] if not a.startswith("--")] or DEFAULT_UNIVERSE

    headers = alpaca_headers()
    equity = get_equity(headers)
    is_open = market_is_open(headers)

    print(f"RomyBot 30-min pivot | equity ${equity:,.0f} | "
          f"risk {RISK_FRACTION:.0%}/trade | market {'OPEN' if is_open else 'CLOSED'}"
          f"{' | DRY-RUN' if dry_run else ''}{' | FINE-5m' if use_fine else ''}")
    print(f"Universe: {', '.join(tickers)}\n")

    placed = skipped = failed = 0
    for t in tickers:
        row = process(t, headers, equity, dry_run, use_fine, is_open)
        d = row["decision"]
        if d == "VALID" and row.get("order_id"):
            placed += 1
            print(f"  [PLACED]  {t}: {row['qty']} sh @ stop {row['entry']} "
                  f"(stop {row['stop']}, tp {row['take_profit']}) "
                  f"order {row['order_id']}")
        elif d == "VALID" and row.get("status") == "DRY_RUN":
            print(f"  [WOULD]   {t}: {row['qty']} sh @ {row['entry']} "
                  f"(stop {row['stop']}, tp {row['take_profit']}) — {row.get('notes','')}")
        elif d in ("ORDER_FAILED", "ERROR"):
            failed += 1
            print(f"  [FAIL]    {t}: {row.get('notes','')}")
        else:
            skipped += 1
            print(f"  [skip]    {t}: {row.get('notes','')}")

    print(f"\nDone. placed={placed} skipped={skipped} failed={failed}")
    print(f"Log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
