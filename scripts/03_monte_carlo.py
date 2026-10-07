#!/usr/bin/env python3
"""Monte Carlo pricing: convergence, variance reduction, and a payoff with no formula.

    python scripts/03_monte_carlo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from volengine import plots                                  # noqa: E402
from volengine.black_scholes import bs_price                 # noqa: E402
from volengine.monte_carlo import (convergence_study,        # noqa: E402
                                   geometric_asian_call, mc_asian_call,
                                   mc_european, mc_pathwise_delta)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def hdr(t: str) -> None:
    print("\n" + "=" * 84)
    print(f"  {t}")
    print("=" * 84)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    S, K, T, r, q, v = 100.0, 105.0, 1.0, 0.042, 0.013, 0.25
    exact = float(bs_price(S, K, T, r, v, q, "call"))
    out = {"exact": exact}

    # ======================================================================
    hdr("1. DOES THE SIMULATION AGREE WITH THE FORMULA?")
    # ======================================================================
    print(f"""
  S={S} K={K} T={T} r={r:.1%} q={q:.1%} sigma={v:.0%}
  Black-Scholes exact price:  {exact:.8f}

  Under the risk-neutral measure the terminal price is EXACTLY

      S_T = S exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z ),   Z ~ N(0,1)

  so for a payoff that depends only on S_T there is no time-stepping error at
  all -- one normal draw per path, and the only error is statistical.
""")
    print(f"  {'method':<24}{'price':>12}{'std err':>11}{'95% CI':>26}"
          f"{'|err|/se':>10}{'sec':>7}")
    print("  " + "-" * 92)
    variants = [("plain", False, False), ("antithetic", True, False),
                ("control variate", False, True), ("antithetic + control", True, True)]
    res = {}
    for label, a, cvar in variants:
        m = mc_european(S, K, T, r, v, q, "call", 500_000, a, cvar, seed=42)
        res[label] = m
        print(f"  {label:<24}{m.price:>12.6f}{m.stderr:>11.6f}"
              f"   [{m.ci_low:.6f}, {m.ci_high:.6f}]"
              f"{abs(m.price-exact)/m.stderr:>10.2f}{m.seconds:>7.2f}")
    print(f"""
  |err|/se is the number to read. If the estimator is unbiased it should be
  under 2 about 95% of the time. Anything persistently above 3 means a BUG,
  not bad luck -- a wrong drift, a wrong discount factor, or a biased sampler.
""")

    # ======================================================================
    hdr("2. VARIANCE REDUCTION: how much, and does it match the theory?")
    # ======================================================================
    base = res["plain"].variance
    print(f"  {'method':<24}{'variance':>14}{'reduction':>12}{'rho':>10}"
          f"{'theory 1/(1-rho^2)':>21}")
    print("  " + "-" * 82)
    for label, m in res.items():
        rho = m.control_corr
        theory = 1.0 / (1.0 - rho ** 2) if np.isfinite(rho) else float("nan")
        print(f"  {label:<24}{m.variance:>14.4f}{base/m.variance:>11.2f}x"
              f"{rho if np.isfinite(rho) else float('nan'):>10.4f}"
              f"{theory if np.isfinite(theory) else float('nan'):>20.2f}x")
    print("""
  ANTITHETIC VARIATES pair every draw Z with -Z. Both are valid N(0,1), so the
  estimator stays unbiased, and the pair's errors partially cancel because they
  are negatively correlated. The gain here is modest, and that is expected:
  max(S_T - K, 0) is CONVEX in Z, and convexity undoes much of the
  cancellation. Antithetics shine on near-linear payoffs.

  CONTROL VARIATES are the real weapon. Alongside the payoff, compute
  something correlated whose exact mean you already know -- here the
  discounted stock price, with E[e^-rT S_T] = S e^-qT. On each path you can
  SEE the error the simulation made on the control, and subtract the
  corresponding part of the payoff's error:

      estimator = mean( payoff + c (S_T - E[S_T]) ),   c* = -Cov/Var

  and the resulting variance is  Var * (1 - rho^2). The 'theory' column above
  is that formula. It matching the observed reduction to two decimal places is
  the check that the implementation is right, not merely helpful.
""")

    # ======================================================================
    hdr("3. CONVERGENCE: the 1/sqrt(N) wall")
    # ======================================================================
    print("""
  The standard error of ANY Monte Carlo estimate falls like s/sqrt(N). Ten
  times the accuracy costs one hundred times the paths. You cannot beat that
  RATE -- variance reduction only shrinks the numerator s, moving the line
  down without changing its slope. That is the single most important practical
  fact about Monte Carlo.
""")
    rows = convergence_study(S, K, T, r, v, q, "call")
    print(f"  {'N paths':>10}" + "".join(f"{lab:>22}" for lab in
                                         [k for k in rows if k not in ("n", "exact")]))
    print("  " + "-" * 100)
    labels = [k for k in rows if k not in ("n", "exact")]
    for i, n in enumerate(rows["n"]):
        line = f"  {n:>10,}"
        for lab in labels:
            line += f"{rows[lab]['abs_error'][i]:>10.5f}+/-{rows[lab]['stderr'][i]:<11.5f}"
        print(line)
    print("\n  (each cell is |MC - exact| +/- standard error)\n")

    n_arr = np.asarray(rows["n"], dtype=float)
    for lab in labels:
        se = np.asarray(rows[lab]["stderr"], dtype=float)
        slope = float(np.polyfit(np.log(n_arr), np.log(se), 1)[0])
        print(f"    {lab:<22} log-log slope of stderr vs N = {slope:+.4f}   "
              f"(theory: -0.5000)")
    print("""
  Every slope should be very close to -0.5. A different slope would mean the
  estimator is not a plain sample mean of i.i.d. draws -- which would be a bug.
""")
    p = plots.plot_mc_convergence(rows, RESULTS)
    print(f"  chart -> {os.path.relpath(p, os.path.join(HERE, '..'))}")

    # ======================================================================
    hdr("4. THE CASE WITH NO FORMULA: an arithmetic-average Asian call")
    # ======================================================================
    print("""
  Payoff = max( average(S_t) - K, 0 ) over 52 weekly observations.

  There is no Black-Scholes formula, because a SUM of lognormals is not
  lognormal. But a sum of NORMALS is normal, so the GEOMETRIC-average version
  IS analytic -- and the two are correlated above 0.999, which makes the
  geometric Asian a near-perfect control variate for the arithmetic one.
  Same paths, one of the two answers known exactly.
""")
    geo = geometric_asian_call(S, K, T, r, v, q, 52)
    a_plain = mc_asian_call(S, K, T, r, v, q, 200_000, 52, False, False, seed=7)
    a_ctrl = mc_asian_call(S, K, T, r, v, q, 200_000, 52, True, True, seed=7)
    print(f"    vanilla European call (exact)        {exact:>12.6f}")
    print(f"    geometric Asian call (exact)         {geo:>12.6f}")
    print(f"    arithmetic Asian, plain MC           {a_plain.price:>12.6f}  "
          f"+/- {a_plain.stderr:.6f}")
    print(f"    arithmetic Asian, controlled MC      {a_ctrl.price:>12.6f}  "
          f"+/- {a_ctrl.stderr:.6f}")
    print(f"""
    correlation with the control     {a_ctrl.control_corr:.6f}
    variance reduction               {a_plain.variance / a_ctrl.variance:.0f}x
    equivalent path saving           you would need {a_plain.variance/a_ctrl.variance:.0f}x
                                     more plain paths for the same accuracy

  Three sanity checks that must hold, and do:
    * Asian < vanilla ({a_ctrl.price:.4f} < {exact:.4f}) -- averaging reduces the
      effective volatility, so the option is cheaper.
    * arithmetic >= geometric ({a_ctrl.price:.4f} >= {geo:.4f}) -- the AM-GM
      inequality, holding path by path.
    * the controlled and plain estimates agree within their error bars.
""")

    # ======================================================================
    hdr("5. GREEKS FROM MONTE CARLO: pathwise vs bumping")
    # ======================================================================
    d = mc_pathwise_delta(S, K, T, r, v, q, 500_000, seed=13)
    print(f"""
    analytic delta                    {d['analytic']:.6f}
    pathwise estimate                 {d['pathwise']:.6f}  +/- {d['pathwise_stderr']:.6f}
    bump with common random numbers   {d['bump_common_random_numbers']:.6f}
    pathwise error                    {d['pathwise_error']:+.6f}   ({abs(d['pathwise_error'])/d['pathwise_stderr']:.2f} standard errors)

  The naive way to get delta is to rerun the whole simulation at S and at S+h.
  Two problems: it costs two simulations, and if you use DIFFERENT random
  numbers for the two runs the Monte Carlo noise swamps the small real
  difference -- your delta is mostly noise. Using the same random numbers
  (common random numbers) fixes most of that.

  The pathwise method is better still: differentiate INSIDE the expectation.
  Since S_T = S exp(...), we have dS_T/dS = S_T/S, so for a call

      delta = e^-rT E[ 1{{S_T > K}} S_T / S ]

  One simulation, no bump, no differencing noise, and it reuses paths you
  already have. The catch is that it needs a payoff differentiable almost
  everywhere -- fine for calls and puts, NOT valid for a digital option whose
  payoff is a step function. Knowing which method applies to which payoff is a
  real interview question.
""")

    out.update({
        "european": {k: dict(price=m.price, stderr=m.stderr, variance=m.variance,
                             corr=m.control_corr, seconds=m.seconds)
                     for k, m in res.items()},
        "convergence": {k: v for k, v in rows.items()},
        "asian": {"geometric_exact": geo, "plain": a_plain.price,
                  "controlled": a_ctrl.price,
                  "plain_stderr": a_plain.stderr, "controlled_stderr": a_ctrl.stderr,
                  "variance_reduction": a_plain.variance / a_ctrl.variance,
                  "control_corr": a_ctrl.control_corr},
        "pathwise_delta": d,
    })
    with open(os.path.join(RESULTS, "monte_carlo.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  data -> results/monte_carlo.json\n")


if __name__ == "__main__":
    main()
