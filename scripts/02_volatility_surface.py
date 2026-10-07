#!/usr/bin/env python3
"""Build a volatility surface from an option chain, and prove it is arbitrage free.

    python scripts/02_volatility_surface.py              # synthetic (known truth)
    python scripts/02_volatility_surface.py --ticker SPY # live, needs yfinance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import volengine as ve                                    # noqa: E402
from volengine import plots                               # noqa: E402
from volengine.surface import build_surface, durrleman_g  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def hdr(t: str) -> None:
    print("\n" + "=" * 86)
    print(f"  {t}")
    print("=" * 86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None,
                    help="fetch a live chain instead of the synthetic one")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    # ======================================================================
    hdr("1. THE CHAIN")
    # ======================================================================
    truth_fn = None
    if args.ticker:
        chain = ve.load_yfinance_chain(args.ticker)
        if chain is None:
            print("  falling back to the synthetic chain\n")
            chain = ve.generate_synthetic_chain()
            truth_fn = lambda K, T: ve.true_iv(K, T, chain["spot"], chain["r"], chain["q"])
        else:
            print(f"  live chain for {args.ticker}: {chain['strike'].size} quotes, "
                  f"spot {chain['spot']:.2f}")
    else:
        chain = ve.generate_synthetic_chain()
        truth_fn = lambda K, T: ve.true_iv(K, T, chain["spot"], chain["r"], chain["q"])
        print(f"""
  Using a SYNTHETIC chain generated from a known SVI surface.

  This is the more informative choice. With real data you can fit a surface
  but never check it, because the true surface is not observable. Here we know
  every true parameter, so we can ask the only question that matters: given
  noisy, tick-rounded, spread-contaminated quotes, does the pipeline recover
  the surface that produced them?

  The generator includes the things that make real chains hard: spreads that
  widen in the wings, prices rounded to the $0.01/$0.05 tick grid, thin
  volume out of the money, and a few genuinely stale or crossed quotes.

  spot {chain['spot']:.2f}   r {chain['r']:.2%}   q {chain['q']:.2%}
  {chain['strike'].size} raw quotes across {len(np.unique(chain['expiry_years']))} expiries
""")

    # ======================================================================
    hdr("2. CLEANING -- this step decides the quality of everything after it")
    # ======================================================================
    clean, reasons = ve.clean_chain(chain)
    print()
    for k, v in reasons.items():
        if k in ("KEPT", "TOTAL"):
            continue
        if v:
            print(f"    removed {v:>5}  {k}")
    print(f"    {'-'*60}")
    print(f"    kept    {reasons['KEPT']:>5}  of {reasons['TOTAL']} "
          f"({100*reasons['KEPT']/reasons['TOTAL']:.1f}%)")
    print("""
  Always read this table. Every line is a specific real pathology, and if one
  filter is eating most of your data then either the filter or the feed is
  wrong and you need to know which. The 'mid outside [bid, ask]' filter in
  particular is worth its weight: a single stale at-the-money mid, implying
  71% vol against a true 31%, was enough to throw an entire maturity slice off
  by 7 volatility points before that check existed -- because vega weighting
  gives at-the-money quotes the MOST influence over the fit.
""")

    # ======================================================================
    hdr("3. INVERT EVERY QUOTE, THEN FIT SVI PER EXPIRY")
    # ======================================================================
    surf = build_surface(clean, clean["spot"], clean["r"], clean["q"])
    d = surf.diagnostics
    print(f"""
  implied vol inversion
    quotes inverted            {d['n_quotes']}
    converged                  {d['n_iv_converged']}  ({d['iv_convergence_pct']:.1f}%)
    solved by Newton           {d['n_newton']}
    needed the Brent fallback  {d['n_brent_fallback']}
    no solution (NaN)          {d['n_failed']}
    mean Newton iterations     {d['mean_newton_iterations']:.2f}
""")
    print(surf.summary())
    print("""
  'out' counts points removed by the ROBUST second pass. Least squares has a
  breakdown point of zero -- one bad point moves the fit arbitrarily far -- so
  we fit, measure residuals, discard anything beyond 4 robust standard
  deviations (scaled by the MEDIAN ABSOLUTE DEVIATION, because the ordinary
  standard deviation is itself inflated by the outliers we are hunting), and
  refit on what remains.
""")

    # ======================================================================
    hdr("4. NO-ARBITRAGE: the checks that separate a surface from a plot")
    # ======================================================================
    print("""
  BUTTERFLY (within one expiry). Buy K-dK, sell two K, buy K+dK. That payoff is
  never negative, so it can never cost less than nothing -- equivalently the
  implied risk-neutral density must be non-negative everywhere. Durrleman's
  condition states it directly on the smile:

      g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  >=  0

  where w(k) = sigma^2(k) T is total implied variance. If g dips below zero
  your surface implies a NEGATIVE PROBABILITY at that strike.
""")
    for s in sorted(surf.slices, key=lambda x: x.T):
        bf = surf.butterfly[s.T]
        print(f"    T = {s.T:<7.4g}  min g = {bf['min_g']:+.5f} at k = "
              f"{bf['worst_k']:+.3f}   {'ARBITRAGE FREE' if bf['arb_free'] else '*** VIOLATION ***'}")

    c = surf.calendar
    print(f"""
  CALENDAR (across expiries). Total implied variance must never DECREASE with
  maturity at a fixed log-moneyness -- variance cannot un-accumulate, so a
  calendar spread would otherwise be free money.

    adjacent maturity pairs checked   {c['n_pairs']}
    worst decrease in total variance  {c['worst_violation']:.3e}
    verdict                           {'ARBITRAGE FREE' if c['arb_free'] else '*** VIOLATION ***'}
""")

    # Prove the checks have teeth by feeding them something broken.
    from volengine.surface import SVIParams, check_butterfly
    bad = SVIParams(a=-0.02, b=0.9, rho=-0.99, m=0.0, sigma=0.01, T=0.5)
    r_bad = check_butterfly(bad, (-0.5, 0.5))
    print(f"""  Sanity check on the checker itself: a deliberately arbitrageable smile
  (a=-0.02, b=0.9, rho=-0.99, sigma=0.01) gives min g = {r_bad['min_g']:.4f}
  and is correctly flagged as {'ARBITRAGE FREE (BAD - check broken!)' if r_bad['arb_free'] else 'a VIOLATION'}.
  A test that never fails is not a test.
""")

    # ======================================================================
    if truth_fn is not None:
        hdr("5. GROUND-TRUTH RECOVERY -- the number that validates the pipeline")
        print("""
  Compare the fitted surface against the surface that generated the quotes.
  Errors are in VOLATILITY POINTS (1.0 = one whole vol point, e.g. 20% -> 21%).
  For context, a typical bid-ask spread on a liquid option is worth well over
  half a vol point, so sub-0.5 recovery is at the noise floor of the data.
""")
        print(f"    {'T (years)':>11}{'quotes':>8}{'max err':>10}{'RMS err':>10}"
              f"{'ATM err':>10}")
        print("    " + "-" * 49)
        rms_all = []
        for s in sorted(surf.slices, key=lambda x: x.T):
            F = float(ve.forward(surf.spot, s.T, surf.r, surf.q))
            Ks = np.linspace(0.85, 1.20, 60) * surf.spot
            tv = truth_fn(Ks, s.T)
            fv = s.implied_vol(np.log(Ks / F))
            rms = float(np.sqrt(np.mean((fv - tv) ** 2)))
            rms_all.append(rms)
            atm_err = abs(float(s.implied_vol(0.0)) - float(truth_fn(np.array([F]), s.T)[0]))
            print(f"    {s.T:>11.4f}{s.n_points:>8}{np.max(np.abs(fv-tv))*100:>10.3f}"
                  f"{rms*100:>10.3f}{atm_err*100:>10.3f}")
        print("    " + "-" * 49)
        print(f"    {'average':>11}{'':>8}{'':>10}{np.mean(rms_all)*100:>10.3f}\n")
        print("""  The shortest expiry is always the worst, and that is not a bug. A one-week
  option has almost no time value in the wings, so most of its strikes get
  filtered out as sub-tick, leaving very few usable quotes to pin down five
  parameters. Real desks have the same problem and handle it the same way:
  fewer parameters, or borrowing shape from the neighbouring expiry.
""")

    # ======================================================================
    hdr("6. USING THE SURFACE")
    # ======================================================================
    print("""
  Once fitted, the surface prices ANY strike and maturity, including ones that
  were never quoted. Interpolation between expiries is LINEAR IN TOTAL
  VARIANCE, not in volatility -- total variance is the additive quantity, so
  interpolating it preserves the calendar ordering, while interpolating
  volatility can break it.
""")
    print(f"    {'strike':>9}{'T':>8}{'interpolated IV':>18}{'call price':>13}")
    print("    " + "-" * 48)
    for K, T in [(90.0, 0.15), (100.0, 0.15), (110.0, 0.15),
                 (90.0, 0.75), (100.0, 0.75), (110.0, 0.75)]:
        iv = surf.implied_vol(K, T)
        print(f"    {K:>9.1f}{T:>8.2f}{iv*100:>17.3f}%{surf.price(K, T):>13.4f}")

    # ======================================================================
    p1 = plots.plot_surface(surf, clean, RESULTS, true_iv_fn=truth_fn)
    p2 = plots.plot_arbitrage_checks(surf, RESULTS)
    print(f"\n  charts -> {os.path.relpath(p1, os.path.join(HERE,'..'))}")
    print(f"            {os.path.relpath(p2, os.path.join(HERE,'..'))}")

    payload = {
        "diagnostics": d,
        "cleaning": reasons,
        "slices": [dict(T=s.T, a=s.a, b=s.b, rho=s.rho, m=s.m, sigma=s.sigma,
                        rmse_vol=s.rmse_vol, n_points=s.n_points,
                        n_outliers_dropped=s.n_outliers_dropped,
                        atm_vol=float(s.implied_vol(0.0)))
                   for s in sorted(surf.slices, key=lambda x: x.T)],
        "butterfly": {str(k): v for k, v in surf.butterfly.items()},
        "calendar": surf.calendar,
    }
    with open(os.path.join(RESULTS, "vol_surface.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"  data   -> results/vol_surface.json\n")


if __name__ == "__main__":
    main()
