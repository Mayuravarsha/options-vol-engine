#!/usr/bin/env python3
"""Pull a live option chain from Yahoo Finance and cache it as a CSV snapshot.

    python scripts/fetch_real_chains.py --ticker SPY
    python scripts/fetch_real_chains.py --ticker AAPL --max-expiries 8

WHY THE EXPIRIES ARE PICKED BY TARGET HORIZON, NOT "THE FIRST N LISTED"
-------------------------------------------------------------------------
SPY lists options expiring on almost every trading day (27 expiries found
when this was written, the first eight all within 20 calendar days). Simply
taking `tk.options[:8]` therefore returns eight near-duplicate SHORT-DATED
slices and nothing further out -- there is no way to draw a "term structure
plot" (implied vol vs maturity) from data that only spans three weeks, and
every one of those slices individually degenerates: annualising a rate or a
dividend yield over a 2-to-6-day window means dividing a tiny, noisy number
by a tiny T, which blows the result up by two orders of magnitude (this was
found empirically -- a first attempt at this script produced implied dividend
yields of -25% on 2-day SPY options, purely from that arithmetic, not from
anything the market actually believes). So instead this script targets a
SPREAD of horizons (`TARGET_DAYS` below: roughly 1 week to 1 year) and picks,
for each target, the nearest expiry actually listed. That is what makes the
resulting chain usable for a term-structure plot and keeps every individual
slice far enough from expiry for its rate/yield to be identifiable at all.

WHY CACHE TO CSV AT ALL
------------------------
An option chain is not a fixed dataset -- it changes every few seconds and the
contracts themselves expire. If the calibration scripts pulled live data every
time, the project would not be reproducible: run it today and run it next
month and you get two different surfaces from two different underlying
datasets, with no record of what actually produced either one.

So this script does the network call exactly once and freezes the result: one
row per contract, with the snapshot date/time written into every row. Every
other script in this project reads the frozen CSV, never the network. That is
what "reproducible offline" means here -- not that the data is fake, but that
it is dated and nailed down.

SCHEMA (one row per contract, denormalised -- ticker/spot/snapshot columns
repeat on every row so the file can be loaded with nothing but the stdlib csv
module, matching load_returns_csv's no-pandas convention elsewhere in data.py):

    ticker, snapshot_date, snapshot_datetime_utc, spot, expiry_date,
    option_type (c/p), strike, bid, ask, last_price, volume, open_interest,
    implied_vol_yahoo

`implied_vol_yahoo` is Yahoo's own IV estimate, included only so the pipeline
here can be cross-checked against an independent source later -- it is never
read by build_surface.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data", "chains")

FIELDS = ["ticker", "snapshot_date", "snapshot_datetime_utc", "spot",
          "expiry_date", "option_type", "strike", "bid", "ask", "last_price",
          "volume", "open_interest", "implied_vol_yahoo"]

# Roughly 1 week to 1 year, log-ish spaced so the short end (where the smile
# moves fastest) is sampled more densely than the long end.
TARGET_DAYS = [7, 14, 30, 60, 90, 180, 270, 365]


def _pick_expiries(all_expiries: Sequence[str], today: dt.date,
                   target_days: Sequence[int]) -> list:
    """For each target horizon, keep the nearest listed expiry that is at
    least 2 days out (anything closer is nearly worthless time value and
    numerically unstable -- see the module docstring). Never picks the same
    expiry twice for two different targets."""
    dated = [(e, (dt.datetime.strptime(e, "%Y-%m-%d").date() - today).days)
             for e in all_expiries]
    dated = [(e, d) for e, d in dated if d >= 2]
    chosen: dict = {}
    for td in target_days:
        e, d = min(dated, key=lambda ed: abs(ed[1] - td))
        chosen.setdefault(e, d)
    return sorted(chosen, key=lambda e: chosen[e])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--max-expiries", type=int, default=8)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. `pip install --user yfinance`")
        sys.exit(1)

    ticker = args.ticker.upper()
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        print(f"no price history for {ticker}; aborting")
        sys.exit(1)
    spot = float(hist["Close"].iloc[-1])

    snap_dt_utc = dt.datetime.now(dt.timezone.utc)
    today = snap_dt_utc.date()
    snap_date = today.isoformat()

    if not tk.options:
        print(f"no listed expiries for {ticker}; aborting")
        sys.exit(1)
    expiries = _pick_expiries(tk.options, today, TARGET_DAYS[:args.max_expiries])
    print(f"  {ticker}: {len(tk.options)} expiries listed, picked "
          f"{len(expiries)} spanning {[(dt.datetime.strptime(e,'%Y-%m-%d').date()-today).days for e in expiries]} days out")

    rows = []
    for exp in expiries:
        try:
            oc = tk.option_chain(exp)
        except Exception as exc:
            print(f"  skipping expiry {exp}: {exc}")
            continue
        for df, otype in ((oc.calls, "c"), (oc.puts, "p")):
            for _, row in df.iterrows():
                def g(col, default=0.0):
                    v = row.get(col, default)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        return default
                    return default if v != v else v  # NaN != NaN
                rows.append({
                    "ticker": ticker,
                    "snapshot_date": snap_date,
                    "snapshot_datetime_utc": snap_dt_utc.isoformat(),
                    "spot": spot,
                    "expiry_date": exp,
                    "option_type": otype,
                    "strike": float(g("strike")),
                    "bid": float(g("bid")),
                    "ask": float(g("ask")),
                    "last_price": float(g("lastPrice")),
                    "volume": int(g("volume", 0.0)),
                    "open_interest": int(g("openInterest", 0.0)),
                    "implied_vol_yahoo": float(g("impliedVolatility", float("nan"))),
                })

    if not rows:
        print(f"no option rows returned for {ticker}; aborting")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{ticker}_{snap_date}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_exp = len(set(r["expiry_date"] for r in rows))
    print(f"{ticker}: spot {spot:.2f}, {len(rows)} contracts across {n_exp} "
          f"expiries -> {os.path.relpath(out_path, os.path.join(HERE, '..'))}")


if __name__ == "__main__":
    main()
