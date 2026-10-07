#!/usr/bin/env python3
"""Independent validation report: exactly how accurate is this pricer?

    python scripts/00_validate_pricer.py

This is a REPORTING script, not a test suite -- tests/test_all.py already
turns every one of these checks into a pass/fail assertion (263 of them).
What this script adds is the number a reader actually wants for a writeup:
not "passed", but the worst error observed, in the units a trader would read
it in (vol points, dollars, relative error), written to
results/validation_report.json so results/RESULTS.md can cite it directly.

FOUR INDEPENDENT CHECKS
------------------------------------------------------------------------
1. PUT-CALL PARITY. C - P = S e^{-qT} - K e^{-rT}. Pure no-arbitrage: true
   for ANY pricing model, not just Black-Scholes, so if this fails the bug is
   in the formula's algebra, not in a modelling assumption.

2. GREEKS VS FINITE DIFFERENCE. The analytic Greeks (closed-form derivatives
   of the BS formula) are checked against a numerically DIFFERENT
   computation: perturb an input by a small h and measure the price's
   response directly. Two unrelated derivations of the same numbers agreeing
   is real evidence; one formula checked against itself is not.

3. IMPLIED VOL ROUND TRIP. Price a known sigma, invert the price, recover
   sigma. This is the only check here that is circular in a mathematical
   sense (BS forward, then BS backward) -- what it actually tests is not
   "is Black-Scholes right" but "does the Newton/Brent inversion in
   implied_vol.py correctly undo what black_scholes.py does", which is a
   real, separate piece of code that can and does have its own bugs (see
   implied_vol.py's docstring on the price-space-vs-vol-space convergence
   bug this project hit during development).

4. INDEPENDENT NUMERICAL-INTEGRATION CROSS-CHECK. The closed-form price is
   S e^{-qT} N(d1) - K e^{-rT} N(d2). But the DEFINITION of a risk-neutral
   price is simpler and does not reference d1/d2 at all:

       price = e^{-rT} * E^Q[ payoff(S_T) ]
             = e^{-rT} * integral of  payoff(S_T) * lognormal_pdf(S_T)  dS_T

   under S_T ~ lognormal with the risk-neutral drift (r - q - sigma^2/2)T.
   This script computes that integral directly with scipy.integrate.quad --
   no d1, no d2, no N(), a genuinely different code path -- and compares it
   to bs_price(). This is deliberately NOT a hardcoded textbook reference
   value: a mistyped citation would silently defeat the entire point of an
   "independent" check, and would be indistinguishable from a real bug in
   whoever is reading this report. A self-contained computation that must
   agree with the closed form by the very definition of what "price" means
   cannot be wrong for that reason, only be implemented wrong -- which is a
   bug worth finding either way.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from volengine.black_scholes import (bs_greeks, bs_price,  # noqa: E402
                                     finite_difference_greeks,
                                     put_call_parity_gap)
from volengine.implied_vol import implied_vol  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def hdr(t: str) -> None:
    print("\n" + "=" * 86)
    print(f"  {t}")
    print("=" * 86)


def bs_price_by_integration(S, K, T, r, sigma, q, option) -> float:
    """The price as the definition says it is: a discounted expectation,
    computed by literally integrating over the risk-neutral density -- no
    d1, no d2, no closed form. See module docstring, check 4."""
    T = max(T, 1e-8)
    sd = max(sigma * np.sqrt(T), 1e-8)
    mu = np.log(S) + (r - q - 0.5 * sigma ** 2) * T
    if option in ("c", "call"):
        integrand = lambda x: max(np.exp(x) - K, 0.0) * norm.pdf(x, mu, sd)
    else:
        integrand = lambda x: max(K - np.exp(x), 0.0) * norm.pdf(x, mu, sd)
    val, _ = quad(integrand, mu - 12 * sd, mu + 12 * sd, limit=200)
    return float(np.exp(-r * T) * val)


def main() -> None:
    t0 = time.perf_counter()
    report: dict = {}

    # ======================================================================
    hdr("1. PUT-CALL PARITY  (model-free no-arbitrage identity)")
    # ======================================================================
    max_gap, n = 0.0, 0
    for S in (50.0, 80.0, 100.0, 130.0, 200.0):
        for K in (50.0, 80.0, 100.0, 130.0, 200.0, 300.0):
            for T in (0.02, 0.1, 0.5, 1.0, 3.0, 7.0):
                for r in (0.0, 0.02, 0.05, 0.09):
                    for q in (0.0, 0.01, 0.03):
                        for sigma in (0.05, 0.2, 0.5, 1.0):
                            n += 1
                            c = bs_price(S, K, T, r, sigma, q, "call")
                            p = bs_price(S, K, T, r, sigma, q, "put")
                            gap = abs(put_call_parity_gap(c, p, S, K, T, r, q))
                            max_gap = max(max_gap, gap)
    print(f"    {n} (S,K,T,r,q,sigma) configurations")
    print(f"    max |C - P - (S e^-qT - K e^-rT)| = {max_gap:.3e}")
    report["put_call_parity"] = {"max_abs_gap": max_gap, "n_configs": n}

    # ======================================================================
    hdr("2. GREEKS VS FINITE DIFFERENCE  (two independent derivations)")
    # ======================================================================
    max_rel = {g: 0.0 for g in ("delta", "gamma", "vega", "theta", "rho")}
    n = 0
    for S in (70.0, 100.0, 135.0):
        for T in (0.05, 0.25, 1.0, 3.0):
            for sigma in (0.1, 0.3, 0.7):
                for option in ("call", "put"):
                    n += 1
                    a = bs_greeks(S, 100.0, T, 0.03, sigma, 0.015, option)
                    f = finite_difference_greeks(S, 100.0, T, 0.03, sigma, 0.015, option)
                    for g in max_rel:
                        scale = max(abs(f[g]), 1e-3)
                        max_rel[g] = max(max_rel[g], abs(a[g] - f[g]) / scale)
    print(f"    {n} (S,T,sigma,option) configurations")
    for g, v in max_rel.items():
        print(f"    max relative error, {g:<6}: {v:.3e}")
    report["greeks_vs_finite_difference"] = {"max_rel_error": max_rel, "n_configs": n}

    # ======================================================================
    hdr("3. IMPLIED VOL ROUND TRIP  (price forward, invert backward)")
    # ======================================================================
    worst, n_fail, n, iters = 0.0, 0, 0, []
    for S in (100.0,):
        for K in (55, 70, 85, 100, 115, 135, 165, 220):
            for T in (0.02, 0.1, 0.5, 1.0, 3.0, 6.0):
                for true_v in (0.06, 0.15, 0.30, 0.6, 1.2):
                    option = "call" if K >= S else "put"
                    px = float(bs_price(S, K, T, 0.03, true_v, 0.01, option))
                    if px < 1e-10:
                        continue
                    n += 1
                    res = implied_vol(px, S, K, T, 0.03, 0.01, option)
                    if not res["converged"]:
                        n_fail += 1
                        continue
                    worst = max(worst, abs(res["iv"] - true_v))
                    iters.append(res["iterations"])
    print(f"    {n} round trips, {n_fail} honestly reported as un-invertible")
    print(f"    worst |recovered vol - true vol| = {worst:.3e}")
    print(f"    mean iterations to converge      = {np.mean(iters):.2f}")
    report["implied_vol_roundtrip"] = {
        "max_abs_error": worst, "n_configs": n, "n_unconverged": n_fail,
        "mean_iterations": float(np.mean(iters)),
    }

    # ======================================================================
    hdr("4. INDEPENDENT CROSS-CHECK: closed form vs numerical integration")
    # ======================================================================
    max_abs, max_rel_cc, n = 0.0, 0.0, 0
    for S in (60.0, 100.0, 140.0):
        for K in (60.0, 100.0, 140.0, 200.0):
            for T in (0.05, 0.5, 2.0):
                for sigma in (0.15, 0.45, 0.9):
                    for r, q in ((0.0, 0.0), (0.04, 0.013)):
                        for option in ("call", "put"):
                            n += 1
                            closed = float(bs_price(S, K, T, r, sigma, q, option))
                            integ = bs_price_by_integration(S, K, T, r, sigma, q, option)
                            abs_err = abs(closed - integ)
                            rel_err = abs_err / max(closed, 1e-4)
                            max_abs = max(max_abs, abs_err)
                            max_rel_cc = max(max_rel_cc, rel_err)
    print(f"    {n} (S,K,T,r,q,sigma,option) configurations")
    print(f"    closed-form price vs quad() integration of e^-rT E^Q[payoff]")
    print(f"    max absolute difference = {max_abs:.3e}")
    print(f"    max relative difference = {max_rel_cc:.3e}")
    report["independent_integration_crosscheck"] = {
        "max_abs_diff": max_abs, "max_rel_diff": max_rel_cc, "n_configs": n,
        "method": "scipy.integrate.quad over the risk-neutral lognormal density, "
                  "no d1/d2/N() -- see module docstring",
    }

    # ======================================================================
    runtime = time.perf_counter() - t0
    report["runtime_seconds"] = runtime
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    hdr("SUMMARY")
    print(f"    runtime: {runtime:.2f}s")
    print(f"    report written to {os.path.relpath(out_path, os.path.join(HERE, '..'))}\n")


if __name__ == "__main__":
    main()
