"""
run_screener.py — Screen a watchlist with RomyBot's swing-trade rules and
place buy-stop bracket orders on Alpaca PAPER for any setup that qualifies.

What it does:
  1. Runs swing_trade.analyze() on each ticker in the universe.
  2. For each VALID setup, computes:
       - entry  : break of the base high (buy-stop trigger)
       - stop   : low of the day (non-negotiable)
       - qty    : risk-based size = floor(RISK_DOLLARS / (entry - stop))
  3. Submits a bracket order on Alpaca paper:
       - buy-stop entry, attached stop-loss, and a 2R take-profit limit.
  4. Logs EVERY evaluated ticker (placed or skipped) to trades_log.csv.

Credentials: pulled from the macOS keychain at runtime (never stored in code).
  alpaca-paper-key    / account alpaca-paper
  alpaca-paper-secret / account alpaca-paper

Usage:
  python run_screener.py                 # default universe
  python run_screener.py NVDA AAPL AMD   # custom tickers
  python run_screener.py --dry-run       # screen + log only, place NO orders
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

import requests

from swing_trade import analyze, Analysis


# ------------------------------ configuration ---------------------------- #

DEFAULT_UNIVERSE = [
    # megacaps
    "NVDA", "AAPL", "MSFT", "AMD", "TSLA",
    "AMZN", "GOOGL", "META", "AVGO", "NFLX",
    # broader liquid names that base more often
    "PLTR", "COIN", "SHOP", "UBER", "CRWD",
    "SMCI", "MU", "QCOM", "NOW", "PANW",
    "SNOW", "DDOG", "NET", "ABNB", "MRVL",
]

ALPACA_BASE = "https://paper-api.alpaca.markets"
RISK_FRACTION = 0.01           # risk 1% of account equity per trade
MAX_POSITION_FRACTION = 0.10   # cap any single position at 10% of equity
TAKE_PROFIT_R = 2.0            # take-profit at 2R
LOG_PATH = os.path.join(os.path.dirname(__file__), "trades_log.csv")

LOG_FIELDS = [
    "timestamp", "ticker", "decision", "score", "price", "entry", "stop",
    "take_profit", "qty", "risk_per_share", "risk_dollars",
    "order_id", "status", "notes",
]


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
    # Cap notional at MAX_POSITION_FRACTION of equity so a tight stop can't
    # produce an oversized position.
    max_notional = equity * MAX_POSITION_FRACTION
    qty_by_notional = math.floor(max_notional / entry)
    qty = min(qty, qty_by_notional)
    return qty, risk_per_share, risk_dollars


def log_row(row: dict) -> None:
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def process(ticker: str, headers: dict, equity: float, dry_run: bool) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"timestamp": now, "ticker": ticker}

    try:
        a: Analysis = analyze(ticker)
    except Exception as exc:  # noqa: BLE001
        row = {**base, "decision": "ERROR", "notes": str(exc)}
        log_row(row)
        return row

    if not a.eligible:
        row = {
            **base, "decision": "SKIP", "score": a.score,
            "price": f"{a.price:.2f}",
            "notes": "; ".join(a.reasons) or "did not qualify",
        }
        log_row(row)
        return row

    entry = max(a.base_high, a.price)
    stop = a.day_low
    qty, rps, risk_dollars = size_position(equity, entry, stop)
    take_profit = entry + TAKE_PROFIT_R * rps

    row = {
        **base, "decision": "VALID", "score": a.score, "price": f"{a.price:.2f}",
        "entry": f"{entry:.2f}", "stop": f"{stop:.2f}",
        "take_profit": f"{take_profit:.2f}", "qty": qty,
        "risk_per_share": f"{rps:.2f}", "risk_dollars": f"{risk_dollars:.0f}",
    }

    if qty < 1:
        row["decision"] = "SKIP"
        row["notes"] = "position size < 1 share at risk budget"
        log_row(row)
        return row

    if dry_run:
        row["status"] = "DRY_RUN"
        row["notes"] = "dry-run: order not submitted"
        log_row(row)
        return row

    resp = submit_bracket(headers, ticker, qty, entry, stop, take_profit)
    if "error" in resp:
        row["decision"] = "ORDER_FAILED"
        row["status"] = "error"
        row["notes"] = resp["error"]
    else:
        row["order_id"] = resp.get("id", "")
        row["status"] = resp.get("status", "")
        row["notes"] = "bracket buy-stop submitted"
    log_row(row)
    return row


# ----------------------------------- main -------------------------------- #

def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    tickers = [a for a in argv[1:] if not a.startswith("--")] or DEFAULT_UNIVERSE

    headers = alpaca_headers()
    equity = get_equity(headers)
    is_open = market_is_open(headers)

    print(f"RomyBot screener | equity ${equity:,.0f} | "
          f"risk {RISK_FRACTION:.0%}/trade | market {'OPEN' if is_open else 'CLOSED'}"
          f"{' | DRY-RUN' if dry_run else ''}")
    print(f"Universe: {', '.join(tickers)}\n")

    placed = skipped = failed = 0
    for t in tickers:
        row = process(t, headers, equity, dry_run)
        d = row["decision"]
        if d == "VALID" and row.get("order_id"):
            placed += 1
            print(f"  [PLACED]  {t}: {row['qty']} sh @ stop {row['entry']} "
                  f"(stop-loss {row['stop']}, tp {row['take_profit']}) "
                  f"order {row['order_id']}")
        elif d == "VALID" and row.get("status") == "DRY_RUN":
            print(f"  [WOULD]   {t}: {row['qty']} sh @ {row['entry']} "
                  f"(stop {row['stop']}, tp {row['take_profit']})")
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
