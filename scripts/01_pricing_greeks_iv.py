#!/usr/bin/env python3
"""Black-Scholes pricing, the Greeks, and inverting to implied volatility.

    python scripts/01_pricing_greeks_iv.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from volengine import plots                                     # noqa: E402
from volengine.black_scholes import (bs_greeks, bs_price,        # noqa: E402
                                     finite_difference_greeks, forward,
                                     put_call_parity_gap)
from volengine.implied_vol import (implied_vol, implied_vol_newton,  # noqa: E402
                                   intrinsic_bounds, manaster_koller_start)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    S, K, T, r, q, v = 100.0, 100.0, 1.0, 0.042, 0.013, 0.22
    out = {}

    # ======================================================================
    hdr("1. ONE OPTION, PRICED AND DISSECTED")
    # ======================================================================
    print(f"""
  Spot S              {S:.2f}
  Strike K            {K:.2f}
  Time to expiry T    {T:.2f} years
  Risk-free rate r    {r:.3%}
  Dividend yield q    {q:.3%}
  Volatility sigma    {v:.2%}
  Forward F = S e^(r-q)T = {float(forward(S, T, r, q)):.4f}
""")
    c = float(bs_price(S, K, T, r, v, q, "call"))
    p = float(bs_price(S, K, T, r, v, q, "put"))
    print(f"  CALL  {c:.6f}      PUT  {p:.6f}")
    gap = float(put_call_parity_gap(c, p, S, K, T, r, q))
    print(f"""
  Put-call parity check:  C - P - (S e^-qT - K e^-rT) = {gap:.3e}
  This is a pure no-arbitrage identity -- it holds in ANY model. A non-zero
  value here means the CODE is wrong, so it is the first thing to check.
""")

    gc = bs_greeks(S, K, T, r, v, q, "call")
    gp = bs_greeks(S, K, T, r, v, q, "put")
    print(f"  {'Greek':<8}{'call':>14}{'put':>14}   what it means")
    print("  " + "-" * 76)
    meaning = {
        "delta": "$ change per $1 move in spot; also the hedge ratio",
        "gamma": "how fast delta moves; the cost of re-hedging",
        "vega": "$ change per 1.00 of vol (divide by 100 for a vol point)",
        "theta": "$ per YEAR of decay (divide by 365 for per day)",
        "rho": "$ change per 1.00 of interest rate",
        "vanna": "how delta moves when vol moves",
        "volga": "how vega moves when vol moves (vol convexity)",
    }
    for g in ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]:
        print(f"  {g:<8}{float(gc[g]):>14.6f}{float(gp[g]):>14.6f}   {meaning[g]}")

    print(f"""
  Reading these in practical units:
    * delta {float(gc['delta']):.3f}  -> hedge one call by shorting {float(gc['delta'])*100:.0f} shares
    * vega  {float(gc['vega']):.2f}   -> one vol POINT (1%) is worth ${float(gc['vega'])/100:.4f}
    * theta {float(gc['theta']):.2f}  -> ${abs(float(gc['theta']))/365:.4f} of decay per calendar day
    * gamma {float(gc['gamma']):.4f} -> after a $1 move, delta changes by {float(gc['gamma']):.4f}
""")

    # ======================================================================
    hdr("2. ARE THE ANALYTIC GREEKS CORRECT? (finite-difference check)")
    # ======================================================================
    print("""
  A typo in a closed-form Greek is invisible: the number looks plausible and
  is simply wrong. So we recompute every Greek by bumping the inputs and
  differencing the PRICE, which is slow but almost impossible to get wrong.
""")
    fd = finite_difference_greeks(S, K, T, r, v, q, "call")
    print(f"  {'Greek':<8}{'analytic':>16}{'finite difference':>20}{'rel error':>14}")
    print("  " + "-" * 60)
    for g in ["delta", "gamma", "vega", "theta", "rho"]:
        a, f = float(gc[g]), float(fd[g])
        print(f"  {g:<8}{a:>16.10f}{f:>20.10f}{abs(a-f)/max(abs(f),1e-9):>14.2e}")

    # ======================================================================
    hdr("3. HOW THE GREEKS BEHAVE  (this is the intuition to internalise)")
    # ======================================================================
    S_grid = np.linspace(55, 150, 300)
    by_T = {}
    for TT in (0.05, 0.25, 1.0, 3.0):
        by_T[TT] = bs_greeks(S_grid, K, TT, r, v, q, "call")
    path = plots.plot_greeks(S_grid, by_T, RESULTS, K=K)
    print(f"""
  chart -> {os.path.relpath(path, os.path.join(HERE, '..'))}

  The four facts that chart teaches:

  1. GAMMA EXPLODES NEAR EXPIRY, AT THE MONEY. The 0.05-year line spikes at
     K and is flat elsewhere. A one-week at-the-money option flips from
     delta 0 to delta 1 over a tiny price range. Being short that gamma is
     how people blow up.

  2. VEGA GROWS WITH MATURITY. Volatility needs TIME to matter, so the
     3-year line dominates. Long-dated options are the vol instrument;
     short-dated options are the gamma instrument.

  3. THETA AND GAMMA ARE THE SAME TRADE, MIRRORED. Wherever gamma is large,
     theta is large and negative. You are paid theta for being short gamma
     and you pay theta for being long it. That is the fundamental trade-off
     in options, and it is visible in these two panels.

  4. DELTA IS A SMOOTHED STEP FUNCTION. As T -> 0 it becomes the indicator
     1{{S > K}}. Longer maturity blurs it, because there is more time for the
     stock to cross the strike.
""")
    for TT in (0.05, 1.0):
        g = bs_greeks(S, K, TT, r, v, q, "call")
        print(f"    ATM, T={TT:>4}y :  gamma {float(g['gamma']):.5f}   "
              f"vega {float(g['vega']):7.3f}   theta/day {float(g['theta'])/365:8.4f}")

    # ======================================================================
    hdr("4. IMPLIED VOLATILITY: RUNNING THE FORMULA BACKWARDS")
    # ======================================================================
    print("""
  Volatility is the only Black-Scholes input nobody can observe -- and the
  PRICE is on the screen. So invert: what sigma makes the model reproduce the
  market price? That number is the implied volatility.

  It works because the price is strictly increasing in sigma, so the root is
  unique. Newton-Raphson uses the derivative, which is vega:

      sigma <- sigma - (model_price - market_price) / vega
""")
    target_v = 0.2735
    px = float(bs_price(S, K, T, r, target_v, q, "call"))
    print(f"  A call quoted at ${px:.6f}. What volatility does that imply?\n")

    sigma = manaster_koller_start(S, K, T, r, q)
    print(f"  {'iter':>5}{'sigma':>12}{'model price':>15}{'error':>14}{'vega':>12}")
    print("  " + "-" * 60)
    print(f"  {'start':>5}{sigma:>12.8f}{float(bs_price(S,K,T,r,sigma,q,'call')):>15.8f}"
          f"{float(bs_price(S,K,T,r,sigma,q,'call'))-px:>14.2e}"
          f"{float(bs_greeks(S,K,T,r,sigma,q,'call')['vega']):>12.4f}")
    path_errs = []
    for i in range(1, 7):
        model = float(bs_price(S, K, T, r, sigma, q, "call"))
        vega = float(bs_greeks(S, K, T, r, sigma, q, "call")["vega"])
        diff = model - px
        path_errs.append(diff)
        sigma = sigma - diff / vega
        m2 = float(bs_price(S, K, T, r, sigma, q, "call"))
        print(f"  {i:>5}{sigma:>12.8f}{m2:>15.8f}{m2-px:>14.2e}{vega:>12.4f}")
    print(f"""
  Recovered {sigma:.10f} against a true {target_v}. Notice the error column:
  each step roughly SQUARES the previous error -- that is quadratic
  convergence, and it is why Newton is the method of choice here.
""")

    # ---- where Newton breaks ---------------------------------------------
    print("  WHERE IT BREAKS, AND WHY THE FALLBACK EXISTS\n")
    money = np.linspace(0.30, 2.60, 70)
    vegas, iters, failed = [], [], []
    for mny in money:
        KK = S * mny
        opt = "call" if KK >= S else "put"
        # Short maturity + extreme strike is the hardest regime there is.
        pxx = float(bs_price(S, KK, 0.05, r, 0.22, q, opt))
        vg = float(bs_greeks(S, KK, 0.05, r, 0.22, q, opt)["vega"])
        vegas.append(vg)
        nw = implied_vol_newton(pxx, S, KK, 0.05, r, q, opt)
        if nw["converged"]:
            iters.append(nw["iterations"])
        else:
            iters.append(0)
            failed.append(mny)
    print(f"    T = 0.05 years (about 2.5 weeks), true vol 22%")
    print(f"    vega at K/S = {money[0]:.2f} : {vegas[0]:.3e}")
    print(f"    vega at K/S = 1.00      : {max(vegas):.3e}")
    print(f"    vega at K/S = {money[-1]:.2f} : {vegas[-1]:.3e}")
    print(f"    ratio between the ATM vega and the worst wing vega: "
          f"{max(vegas)/max(min(vegas), 1e-300):.3e}")
    print(f"\n    Newton alone failed at {len(failed)} of {len(money)} strikes"
          + (f" (K/S = {', '.join(f'{x:.2f}' for x in failed[:8])}"
             + (" ...)" if len(failed) > 8 else ")") if failed else ""))
    print("""
    Vega is the DENOMINATOR of the Newton step. Across this chain it spans
    many orders of magnitude, and out in the wings it collapses toward zero:
    the step blows up and the iterate leaves the valid range. Brent's method
    brackets the root -- it keeps a lower and an upper bound that straddle the
    answer -- so it cannot diverge. Slower, but it always answers.

    Two guards in this implementation matter as much as the fallback itself:
      * the Manaster-Koller start begins where vega is MAXIMISED, which is the
        most numerically stable place to be;
      * convergence is measured in VOLATILITY space (the size of the Newton
        step), not in price space. An absolute price tolerance declares
        success while the vol is still wrong in its second decimal on a
        low-vega wing option -- a bug this project actually had, worth 1.1
        vol points of silent error until the criterion was changed.
""")

    lo, hi = intrinsic_bounds(S, K, T, r, q, "call")
    bad = implied_vol(hi + 1.0, S, K, T, r, q, "call")
    print(f"    No-arbitrage bounds for this call: [{lo:.4f}, {hi:.4f}]")
    print(f"    Asking for a price of {hi+1.0:.4f} returns iv = {bad['iv']} "
          f"({bad['reason']})")
    print("    Returning NaN rather than a number is the correct behaviour: no")
    print("    volatility reproduces an arbitrageable price, and quietly")
    print("    inventing one is how broken surfaces get built.\n")

    path = plots.plot_iv_convergence({
        "paths": {"ATM call": path_errs},
        "moneyness": money.tolist(),
        "vega": vegas,
        "iterations": iters,
        "failed_moneyness": failed,
    }, RESULTS)
    print(f"  chart -> {os.path.relpath(path, os.path.join(HERE, '..'))}\n")

    out = {"call": c, "put": p, "parity_gap": gap,
           "greeks_call": {k: float(np.asarray(vv).item()) for k, vv in gc.items()},
           "greeks_put": {k: float(np.asarray(vv).item()) for k, vv in gp.items()},
           "iv_recovered": sigma, "iv_true": target_v,
           "newton_errors": [float(x) for x in path_errs]}
    with open(os.path.join(RESULTS, "pricing_greeks_iv.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
