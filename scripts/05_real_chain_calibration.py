#!/usr/bin/env python3
"""Calibrate an SVI surface to a REAL, cached option chain and report every
number needed to defend the result: forward/rate extraction, cleaning-filter
counts, fit RMSE per expiry, arbitrage-check verdicts, and fit runtime.

    python scripts/05_real_chain_calibration.py --ticker SPY
    python scripts/05_real_chain_calibration.py --ticker AAPL --r-assumed 0.043

Reads data/chains/{TICKER}_*.csv (see scripts/fetch_real_chains.py -- run
that first if the file does not exist yet; this script never touches the
network). Writes results/vol_surface_{ticker}.json and three PNGs:
results/vol_surface_{ticker}.png, results/smiles_{ticker}.png,
results/term_structure_{ticker}.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import volengine as ve                     # noqa: E402
from volengine import plots                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
CHAINS = os.path.join(ROOT, "data", "chains")


def hdr(t: str) -> None:
    print("\n" + "=" * 86)
    print(f"  {t}")
    print("=" * 86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--r-assumed", type=float, default=0.042,
                    help="risk-free rate used for discounting; see "
                         "implied_forward_rate's docstring for why this is "
                         "fixed rather than also extracted from the chain")
    args = ap.parse_args()
    ticker = args.ticker.upper()
    os.makedirs(RESULTS, exist_ok=True)

    # ======================================================================
    hdr(f"1. LOAD  --  {ticker}")
    # ======================================================================
    matches = sorted(glob.glob(os.path.join(CHAINS, f"{ticker}_*.csv")))
    if not matches:
        print(f"  no cached chain for {ticker} in data/chains/. Run:\n"
              f"    python scripts/fetch_real_chains.py --ticker {ticker}")
        sys.exit(1)
    chain_path = matches[-1]   # most recent snapshot if more than one exists
    raw = ve.load_cached_chain(chain_path)
    n_expiries = int(np.unique(raw["expiry_years"]).size)
    print(f"  {os.path.relpath(chain_path, ROOT)}")
    print(f"  snapshot {raw['snapshot_date']}   spot {raw['spot']:.2f}   "
          f"{raw['strike'].size} contracts across {n_expiries} expiries")

    # ======================================================================
    hdr("2. FORWARD / RATE EXTRACTION  (put-call parity, r held fixed)")
    # ======================================================================
    curve = ve.implied_forward_rate(raw, raw["spot"], assumed_r=args.r_assumed)
    print(f"""
  r is FIXED at {args.r_assumed:.2%} (see implied_forward_rate's docstring:
  short-dated option prices do not carry enough interest-rate signal to
  identify r separately from bid-ask noise -- an earlier attempt at fitting
  both r and the forward F simultaneously produced r estimates swinging from
  -23% to +21% expiry to expiry on this exact SPY chain, while F itself
  stayed put the whole time). What's extracted here, per expiry, is the
  FORWARD (and hence the dividend/borrow yield q), which the chain actually
  can tell you.
""")
    print(f"    {'T (d)':>7}{'F':>10}{'q':>9}{'n_pairs':>9}{'f_std':>8}  method")
    print("    " + "-" * 62)
    for t, c in sorted(curve.items()):
        print(f"    {t*365:>7.0f}{c['F']:>10.2f}{c['q']*100:>8.2f}%{c['n_pairs']:>9}"
              f"{c['f_std']:>8.3f}  {c['method']}")

    # Needed by clean_chain's OTM/moneyness filters; must be injected BEFORE
    # cleaning, and the curve above must be extracted from the RAW chain
    # (before cleaning drops one side of every strike via otm_only=True) --
    # otherwise there are no (call, put) pairs left to regress.
    raw["forward"] = np.array([curve[float(t)]["F"] for t in raw["expiry_years"]])

    # ======================================================================
    hdr("3. CLEAN  --  this step decides the quality of everything after it")
    # ======================================================================
    clean, reasons = ve.clean_chain(raw)
    print()
    for k, v in reasons.items():
        if k in ("KEPT", "TOTAL"):
            continue
        if v:
            print(f"    removed {v:>5}  {k}")
    print(f"    {'-'*60}")
    print(f"    kept    {reasons['KEPT']:>5}  of {reasons['TOTAL']} "
          f"({100*reasons['KEPT']/reasons['TOTAL']:.1f}%)")

    # ======================================================================
    hdr("4. FIT SVI PER EXPIRY")
    # ======================================================================
    r_arr, q_arr = ve.apply_forward_rate(clean, curve)
    t0 = time.perf_counter()
    surf = ve.build_surface(clean, clean["spot"], r_arr, q_arr)
    fit_seconds = time.perf_counter() - t0
    print(f"\n  fit runtime: {fit_seconds:.2f}s for {len(surf.slices)} expiries "
          f"({fit_seconds/max(len(surf.slices),1):.2f}s/expiry)")
    print(surf.summary())
    n_extrap = sum(1 for s in surf.slices if not s.reliable())
    if n_extrap:
        print(f"""
  {n_extrap} slice(s) flagged [extrapolated]: their fitted SVI vertex (m)
  lies outside the strikes actually observed for that expiry. Their RMSE is
  still low (they fit the data they saw), but parameters like rho should not
  be read as "the skew" for these -- see docs/CONCEPTS.md. The economic
  skew reported below uses a fixed-moneyness metric instead, which stays
  meaningful even for a flagged slice.
""")

    # ======================================================================
    hdr("5. ARBITRAGE CHECKS")
    # ======================================================================
    for s in sorted(surf.slices, key=lambda x: x.T):
        bf = surf.butterfly[s.T]
        print(f"    T={s.T*365:>4.0f}d   min g = {bf['min_g']:+.5f}   "
              f"{'ARBITRAGE FREE' if bf['arb_free'] else '*** VIOLATION ***'}")
    c = surf.calendar
    kr = c.get("k_range_checked", (float("nan"), float("nan")))
    print(f"""
    calendar: {'ARBITRAGE FREE' if c['arb_free'] else '*** VIOLATION ***'}
    worst total-variance decrease: {c['worst_violation']:.3e}
    checked over k in [{kr[0]:+.2f}, {kr[1]:+.2f}]  (the intersection of what
    every expiry actually observed -- see build_surface's docstring for why
    a fixed, wide default range is the wrong comparison on a real chain)
""")

    # ======================================================================
    hdr("6. SKEW AND TERM STRUCTURE  (model-robust metric, not raw SVI rho)")
    # ======================================================================
    slice_rows = []
    for s in sorted(surf.slices, key=lambda x: x.T):
        atm = float(s.implied_vol(0.0))
        skew = float(s.implied_vol(-0.10) - s.implied_vol(0.10))
        slice_rows.append(dict(
            T=s.T, days=round(s.T * 365), n_points=s.n_points,
            n_outliers_dropped=s.n_outliers_dropped,
            a=s.a, b=s.b, rho=s.rho, m=s.m, sigma=s.sigma,
            rmse_vol=s.rmse_vol, k_lo=s.k_lo, k_hi=s.k_hi,
            reliable=s.reliable(), atm_vol=atm, skew_10=skew,
        ))
        print(f"    T={s.T*365:>4.0f}d   ATM vol {atm*100:>6.2f}%   "
              f"skew(k=-.10 minus k=+.10) {skew*100:>+6.2f}vp"
              f"{'' if s.reliable() else '   [extrapolated]'}")

    # ======================================================================
    hdr("7. PLOTS AND JSON")
    # ======================================================================
    truth_fn = None
    p1 = plots.plot_surface(surf, clean, RESULTS, name=f"vol_surface_{ticker}.png",
                            true_iv_fn=truth_fn)
    p2 = plots.plot_smiles_grid(surf, clean, RESULTS, name=f"smiles_{ticker}.png",
                                title=ticker)
    atm_curve = {ticker: [(row["T"], row["atm_vol"]) for row in slice_rows]}
    skew_curve = {ticker: [(row["T"], row["skew_10"]) for row in slice_rows]}
    p3 = plots.plot_term_structure(atm_curve, RESULTS, name=f"term_structure_{ticker}.png",
                                   skew_curves=skew_curve)
    for p in (p1, p2, p3):
        print(f"  chart -> {os.path.relpath(p, ROOT)}")

    payload = dict(
        ticker=ticker, chain_file=os.path.relpath(chain_path, ROOT),
        snapshot_date=raw["snapshot_date"], spot=raw["spot"],
        r_assumed=args.r_assumed,
        rate_curve={f"{t:.6f}": c for t, c in sorted(curve.items())},
        cleaning=reasons, diagnostics=surf.diagnostics,
        fit_runtime_seconds=fit_seconds,
        slices=slice_rows,
        butterfly={f"{k:.6f}": v for k, v in surf.butterfly.items()},
        calendar=surf.calendar,
    )
    out_path = os.path.join(RESULTS, f"vol_surface_{ticker}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"  data  -> {os.path.relpath(out_path, ROOT)}\n")


if __name__ == "__main__":
    main()
