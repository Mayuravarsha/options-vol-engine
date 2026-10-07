#!/usr/bin/env python3
"""Validation suite. Every claim the engine makes is checked here.

Run:  python tests/test_all.py

There is no test framework dependency on purpose -- just a CHECK macro -- so
this runs anywhere numpy and scipy run.

WHAT MAKES THESE TESTS WORTH HAVING
-----------------------------------
Most of them are not "does the function return a number". They are IDENTITIES
and RECOVERY tests -- statements that must hold exactly, or estimators asked to
recover parameters from data generated with those very parameters. Those are
the tests that catch real errors:

  * put-call parity must hold to machine precision (pure no-arbitrage, no model)
  * analytic Greeks must match finite differences (independent computation)
  * implied vol must invert Black-Scholes exactly (round trip)
  * Monte Carlo must land within a few standard errors of the closed form
  * the GARCH MLE must recover parameters from simulated GARCH data
  * SVI must recover the surface that generated the synthetic chain
  * the fast filtered GARCH recursion must equal the naive loop, bit for bit
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import volengine as ve                                   # noqa: E402
from volengine import garch as G                         # noqa: E402
from volengine.black_scholes import (bs_greeks, bs_price,  # noqa: E402
                                     finite_difference_greeks,
                                     put_call_parity_gap)
from volengine.implied_vol import (implied_vol, implied_vol_brent,  # noqa: E402
                                   implied_vol_newton, intrinsic_bounds)
from volengine.monte_carlo import (geometric_asian_call, mc_asian_call,  # noqa: E402
                                   mc_european, mc_pathwise_delta)
from volengine.surface import (build_surface, check_butterfly,  # noqa: E402
                               check_calendar, fit_svi_slice)

FAILS = 0
CHECKS = 0


def CHECK(cond, msg: str) -> None:
    global FAILS, CHECKS
    CHECKS += 1
    if not cond:
        FAILS += 1
        print(f"    FAIL  {msg}")


def near(a, b, tol, msg: str) -> None:
    CHECK(abs(float(a) - float(b)) <= tol,
          f"{msg}  (got {float(a):.10g}, want {float(b):.10g}, tol {tol:g})")


def section(name: str) -> None:
    print(f"\n[{name}]")


# ===========================================================================
def test_black_scholes() -> None:
    section("1. Black-Scholes: identities that must hold exactly")
    S, K, T, r, q, v = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20

    # Put-call parity: pure no-arbitrage, must be exact.
    c = bs_price(S, K, T, r, v, q, "call")
    p = bs_price(S, K, T, r, v, q, "put")
    near(put_call_parity_gap(c, p, S, K, T, r, q), 0.0, 1e-12,
         "put-call parity at the money")
    for KK in (60.0, 80.0, 120.0, 160.0):
        c = bs_price(S, KK, T, r, v, q, "call")
        p = bs_price(S, KK, T, r, v, q, "put")
        near(put_call_parity_gap(c, p, S, KK, T, r, q), 0.0, 1e-11,
             f"put-call parity at K={KK}")

    # Monotonic and bounded.
    vols = np.linspace(0.01, 2.0, 200)
    prices = np.array([float(bs_price(S, K, T, r, x, q, "call")) for x in vols])
    CHECK(np.all(np.diff(prices) > 0), "call price is strictly increasing in vol")
    CHECK(prices[-1] < S * np.exp(-q * T) + 1e-9, "call price below S e^{-qT}")

    ks = np.linspace(50, 200, 200)
    cs = np.array([float(bs_price(S, x, T, r, v, q, "call")) for x in ks])
    CHECK(np.all(np.diff(cs) < 0), "call price is decreasing in strike")

    # Degenerate cases collapse to intrinsic value.
    near(bs_price(120.0, 100.0, 0.0, r, v, q, "call"), 20.0, 1e-9, "T=0 call intrinsic")
    near(bs_price(80.0, 100.0, 0.0, r, v, q, "put"), 20.0, 1e-9, "T=0 put intrinsic")
    near(bs_price(120.0, 100.0, 1.0, 0.0, 0.0, 0.0, "call"), 20.0, 1e-9,
         "zero vol, zero rate -> intrinsic")

    # Deep in/out of the money limits.
    CHECK(float(bs_price(100.0, 10000.0, 1.0, r, v, q, "call")) < 1e-8,
          "deep OTM call is worthless")
    near(bs_price(10000.0, 100.0, 1.0, r, v, q, "call"),
         10000 * np.exp(-q) - 100 * np.exp(-r), 1e-4, "deep ITM call = forward value")


def test_greeks() -> None:
    section("2. Greeks: analytic vs finite difference")
    worst = 0.0
    for S in (80.0, 100.0, 125.0):
        for T in (0.08, 0.5, 2.0):
            for v in (0.12, 0.35):
                for opt in ("call", "put"):
                    a = bs_greeks(S, 100.0, T, 0.03, v, 0.01, opt)
                    f = finite_difference_greeks(S, 100.0, T, 0.03, v, 0.01, opt)
                    for g in ("delta", "gamma", "vega", "theta", "rho"):
                        scale = max(abs(f[g]), 1e-3)
                        rel = abs(a[g] - f[g]) / scale
                        worst = max(worst, rel)
                        CHECK(rel < 2e-4,
                              f"{g} S={S} T={T} v={v} {opt}: rel err {rel:.2e}")
    print(f"    worst relative error across 36 configurations: {worst:.2e}")

    # Gamma and vega are identical for calls and puts (parity has neither).
    for S in (85.0, 100.0, 115.0):
        c = bs_greeks(S, 100.0, 0.7, 0.03, 0.25, 0.01, "call")
        p = bs_greeks(S, 100.0, 0.7, 0.03, 0.25, 0.01, "put")
        near(c["gamma"], p["gamma"], 1e-12, f"gamma call==put at S={S}")
        near(c["vega"], p["vega"], 1e-12, f"vega call==put at S={S}")
        near(c["delta"] - p["delta"], np.exp(-0.01 * 0.7), 1e-12,
             f"delta_call - delta_put = e^-qT at S={S}")

    # Gamma is maximised near the money.
    Sg = np.linspace(60, 160, 400)
    gam = np.array([float(bs_greeks(s, 100.0, 0.25, 0.03, 0.2, 0.0)["gamma"]) for s in Sg])
    CHECK(abs(Sg[int(np.argmax(gam))] - 100.0) < 4.0, "gamma peaks near the strike")

    # Long-dated ATM has more vega than short-dated ATM.
    v_short = float(bs_greeks(100, 100, 0.08, 0.03, 0.2, 0.0)["vega"])
    v_long = float(bs_greeks(100, 100, 2.0, 0.03, 0.2, 0.0)["vega"])
    CHECK(v_long > v_short * 3, "vega grows strongly with maturity")


def test_implied_vol() -> None:
    section("3. Implied volatility: round-trip inversion")
    S, r, q = 100.0, 0.03, 0.01
    n_fail = 0
    worst = 0.0
    total = 0
    iters = []
    for K in (60, 75, 90, 100, 110, 130, 160):
        for T in (0.02, 0.1, 0.5, 1.0, 3.0):
            for true_v in (0.08, 0.18, 0.35, 0.9):
                opt = "call" if K >= S else "put"
                px = float(bs_price(S, K, T, r, true_v, q, opt))
                if px < 1e-10:
                    continue
                total += 1
                res = implied_vol(px, S, K, T, r, q, opt)
                if not res["converged"]:
                    n_fail += 1
                    continue
                worst = max(worst, abs(res["iv"] - true_v))
                iters.append(res["iterations"])
    print(f"    {total} round trips, {n_fail} honestly reported as "
          f"un-invertible, worst error {worst:.3e}, "
          f"mean {np.mean(iters):.1f} iterations")
    # The contract is: EITHER recover the volatility accurately, OR say you
    # cannot. Silently returning a wrong number is the only unacceptable
    # outcome, so both branches are tested.
    CHECK(worst < 1e-6, f"every converged inversion is accurate (worst {worst:.2e})")
    CHECK(n_fail <= 3, f"at most a handful of un-invertible wing quotes (got {n_fail})")
    CHECK(np.mean(iters) < 15, "Newton converges in few iterations")

    # Prices outside the no-arbitrage bounds must return NaN, not a number.
    lo, hi = intrinsic_bounds(100.0, 100.0, 1.0, 0.03, 0.01, "call")
    CHECK(np.isnan(implied_vol(hi + 5.0, 100.0, 100.0, 1.0, 0.03, 0.01, "call")["iv"]),
          "price above the upper bound -> NaN")
    CHECK(np.isnan(implied_vol(max(lo - 1.0, -1.0), 100.0, 100.0, 1.0, 0.03, 0.01,
                               "call")["iv"]),
          "price below intrinsic -> NaN")

    # Newton and Brent must agree where both succeed.
    for K in (70, 100, 140):
        px = float(bs_price(100.0, K, 0.5, 0.03, 0.27, 0.01, "call"))
        a = implied_vol_newton(px, 100.0, K, 0.5, 0.03, 0.01, "call")
        b = implied_vol_brent(px, 100.0, K, 0.5, 0.03, 0.01, "call")
        if a["converged"] and b["converged"]:
            near(a["iv"], b["iv"], 1e-6, f"Newton == Brent at K={K}")

    # Deep OTM with tiny vega: the fallback must engage rather than fail.
    px = float(bs_price(100.0, 400.0, 0.05, 0.03, 0.2, 0.01, "call"))
    if px > 1e-14:
        res = implied_vol(px, 100.0, 400.0, 0.05, 0.03, 0.01, "call")
        CHECK(res["converged"] or np.isnan(res["iv"]),
              "extreme wing either converges or honestly returns NaN")


def test_monte_carlo() -> None:
    section("4. Monte Carlo: unbiasedness and variance reduction")
    S, K, T, r, q, v = 100.0, 105.0, 1.0, 0.03, 0.01, 0.25
    exact = float(bs_price(S, K, T, r, v, q, "call"))

    plain = mc_european(S, K, T, r, v, q, "call", 400_000, False, False, seed=1)
    anti = mc_european(S, K, T, r, v, q, "call", 400_000, True, False, seed=1)
    ctrl = mc_european(S, K, T, r, v, q, "call", 400_000, False, True, seed=1)
    both = mc_european(S, K, T, r, v, q, "call", 400_000, True, True, seed=1)

    print(f"    exact Black-Scholes  {exact:.6f}")
    for name, res in [("plain", plain), ("antithetic", anti),
                      ("control", ctrl), ("both", both)]:
        vr = plain.variance / res.variance
        print(f"    {name:<12} {res.price:.6f}  se {res.stderr:.6f}  "
              f"variance reduction {vr:6.1f}x  |err|/se = "
              f"{abs(res.price-exact)/res.stderr:.2f}")
        CHECK(abs(res.price - exact) < 4 * res.stderr,
              f"MC {name} within 4 standard errors of the exact price")

    # The theory says the variance reduction from a control variate is exactly
    # 1/(1 - rho^2). Checking the OBSERVED reduction against that prediction is
    # a far stronger test than "the variance got smaller": it validates the
    # mathematics, not just the direction.
    rho = ctrl.control_corr
    predicted = 1.0 / (1.0 - rho ** 2)
    observed = plain.variance / ctrl.variance
    print(f"    control variate: rho = {rho:.4f} -> theory predicts "
          f"{predicted:.2f}x, observed {observed:.2f}x")
    CHECK(abs(observed - predicted) / predicted < 0.15,
          f"variance reduction matches 1/(1-rho^2) (pred {predicted:.2f}, "
          f"obs {observed:.2f})")
    CHECK(observed > 3.0, "control variate gives a material reduction")
    CHECK(both.variance <= ctrl.variance * 1.05,
          "antithetic + control is no worse than control alone")

    # Standard error must fall like 1/sqrt(N).
    se_small = mc_european(S, K, T, r, v, q, "call", 20_000, False, False, seed=3).stderr
    se_big = mc_european(S, K, T, r, v, q, "call", 320_000, False, False, seed=3).stderr
    ratio = se_small / se_big
    CHECK(3.2 < ratio < 4.8, f"16x paths -> ~4x smaller stderr (got {ratio:.2f}x)")

    # Put too, and check parity holds on the simulated prices.
    mp = mc_european(S, K, T, r, v, q, "put", 400_000, True, True, seed=5)
    exact_p = float(bs_price(S, K, T, r, v, q, "put"))
    CHECK(abs(mp.price - exact_p) < 4 * mp.stderr, "MC put within 4 se")

    # Asian: the geometric control must be extremely effective.
    a_plain = mc_asian_call(S, K, T, r, v, q, 60_000, 52, False, False, seed=9)
    a_ctrl = mc_asian_call(S, K, T, r, v, q, 60_000, 52, True, True, seed=9)
    vr = a_plain.variance / a_ctrl.variance
    print(f"    Asian: plain se {a_plain.stderr:.5f} -> controlled "
          f"{a_ctrl.stderr:.5f}  ({vr:.0f}x variance reduction, "
          f"rho={a_ctrl.control_corr:.5f})")
    CHECK(vr > 20, f"geometric control gives a big reduction (got {vr:.1f}x)")
    CHECK(a_ctrl.price < float(bs_price(S, K, T, r, v, q, "call")),
          "Asian is cheaper than the vanilla (averaging lowers volatility)")
    CHECK(a_ctrl.price > geometric_asian_call(S, K, T, r, v, q, 52),
          "arithmetic Asian >= geometric Asian (AM-GM inequality)")

    # Pathwise delta.
    d = mc_pathwise_delta(S, K, T, r, v, q, 400_000, seed=11)
    CHECK(abs(d["pathwise_error"]) < 4 * d["pathwise_stderr"],
          "pathwise delta matches the analytic delta")
    print(f"    pathwise delta {d['pathwise']:.5f} vs analytic {d['analytic']:.5f}")


def test_garch() -> None:
    section("5. GARCH: fast recursion equivalence and parameter recovery")

    # 5a. The vectorised filter must equal the naive loop exactly.
    def ref(eps, om, al, ga, be, v0):
        n = eps.size
        var = np.empty(n)
        v = v0
        for t in range(n):
            var[t] = v
            e = eps[t]
            v = om + (al + (ga if e < 0 else 0.0)) * e * e + be * v
            if not np.isfinite(v) or v <= 0:
                v = 1e-12
        return var

    rng = np.random.default_rng(0)
    eps = rng.standard_normal(3000) * 0.011
    for ga in (0.0, 0.06):
        a = ref(eps, 2e-6, 0.09, ga, 0.89, 1e-4)
        b = G._variance_path(eps, 2e-6, 0.09, ga, 0.89, 1e-4)
        CHECK(np.allclose(a, b, rtol=0, atol=0),
              f"filtered recursion is bit-identical to the loop (gamma={ga})")

    # 5b. Recover known parameters from simulated data.
    TRUE = dict(omega=2.0e-6, alpha=0.09, beta=0.89, mu=0.0004)
    r = G.simulate_garch(5000, **TRUE, seed=1)
    t0 = time.perf_counter()
    fit = G.fit_garch(r)
    print(f"    GARCH fit on 5,000 observations in {time.perf_counter()-t0:.2f}s")
    CHECK(fit.converged, "GARCH optimisation converged")
    for name, truth in TRUE.items():
        est = getattr(fit, name)
        se = fit.std_errors.get(name, np.nan)
        z = (est - truth) / se if se and np.isfinite(se) and se > 0 else np.nan
        print(f"      {name:<6} true {truth:<11.6g} est {est:<11.6g} z = {z:+.2f}")
        CHECK(abs(z) < 3.0, f"{name} recovered within 3 standard errors")

    CHECK(0.9 < fit.persistence < 1.0, "persistence in the empirically plausible range")
    CHECK(fit.long_run_var > 0, "long-run variance is positive")
    near(fit.long_run_var,
         TRUE["omega"] / (1 - TRUE["alpha"] - TRUE["beta"]),
         TRUE["omega"] / (1 - TRUE["alpha"] - TRUE["beta"]) * 0.35,
         "long-run variance close to the truth")

    # 5c. GJR must find the leverage effect when it is there, and not when it is not.
    rg = G.simulate_garch(5000, omega=2.0e-6, alpha=0.03, beta=0.88,
                          gamma=0.10, mu=0.0003, seed=5)
    fg = G.fit_garch(rg, model="gjr")
    t_gamma = fg.gamma / fg.std_errors.get("gamma", np.nan)
    print(f"      GJR gamma: true 0.10, est {fg.gamma:.4f}, t = {t_gamma:.2f}")
    CHECK(fg.gamma > 0.04 and t_gamma > 3, "GJR detects a real leverage effect")

    fg0 = G.fit_garch(r, model="gjr")     # symmetric data
    t0g = abs(fg0.gamma / fg0.std_errors.get("gamma", np.nan))
    print(f"      GJR on symmetric data: gamma {fg0.gamma:.4f}, |t| = {t0g:.2f}")
    CHECK(t0g < 3.5, "GJR does not invent a leverage effect in symmetric data")

    # 5d. Forecast sanity: multi-step forecasts revert toward the long run.
    fc = G.forecast_variance(fit, last_eps=0.05, last_var=fit.long_run_var * 4, horizon=250)
    CHECK(fc[0] > fc[-1], "forecast decays from a high level")
    CHECK(abs(fc[-1] - fit.long_run_var) < 0.35 * fit.long_run_var,
          "long-horizon forecast approaches the long-run variance")
    CHECK(np.all(np.diff(fc) < 1e-12), "decay is monotone from above")

    # 5e. EWMA is GARCH with omega=0 and persistence exactly 1.
    ew = G.ewma_variance(r, 0.94)
    CHECK(np.all(ew > 0), "EWMA variance is positive")
    CHECK(len(ew) == len(r), "EWMA returns one variance per observation")


def test_surface() -> None:
    section("6. SVI surface: fit quality, arbitrage checks, ground-truth recovery")
    chain = ve.generate_synthetic_chain(seed=2024)
    clean, reasons = ve.clean_chain(chain)
    CHECK(reasons["KEPT"] > 0.6 * reasons["TOTAL"],
          f"cleaning keeps most quotes ({reasons['KEPT']}/{reasons['TOTAL']})")

    surf = build_surface(clean, clean["spot"], clean["r"], clean["q"])
    CHECK(len(surf.slices) >= 5, "fitted at least 5 maturity slices")
    CHECK(surf.diagnostics["iv_convergence_pct"] > 99.0,
          f"IV inversion converged on {surf.diagnostics['iv_convergence_pct']:.1f}% of quotes")

    errs = []
    for s in surf.slices:
        CHECK(s.rmse_vol < 0.02,
              f"T={s.T:.4g} fit RMSE {s.rmse_vol*100:.2f} vol pts is under 2.0")
        bf = surf.butterfly[s.T]
        CHECK(bf["arb_free"], f"T={s.T:.4g} is butterfly arbitrage free")

        F = float(ve.forward(clean["spot"], s.T, clean["r"], clean["q"]))
        Ks = np.linspace(0.85, 1.20, 40) * clean["spot"]
        tv = ve.true_iv(Ks, s.T, clean["spot"], clean["r"], clean["q"])
        fv = s.implied_vol(np.log(Ks / F))
        rms = float(np.sqrt(np.mean((fv - tv) ** 2)))
        errs.append(rms)
        print(f"      T={s.T:<7.4g} fit RMSE {s.rmse_vol*100:5.3f}  "
              f"vs TRUTH {rms*100:5.3f} vol pts  min g {bf['min_g']:+.4f}")
    CHECK(float(np.mean(errs)) < 0.01,
          f"average recovery error {np.mean(errs)*100:.3f} vol pts is under 1.0")
    CHECK(surf.calendar["arb_free"], "no calendar arbitrage across the surface")

    # An SVI slice with a deliberately silly parameter set must be CAUGHT.
    from volengine.surface import SVIParams
    bad = SVIParams(a=-0.02, b=0.9, rho=-0.99, m=0.0, sigma=0.01, T=0.5)
    CHECK(not check_butterfly(bad, (-0.5, 0.5))["arb_free"],
          "the butterfly check actually rejects an arbitrageable smile")

    good = sorted(surf.slices, key=lambda s: s.T)
    bad_cal = [good[-1], SVIParams(a=good[-1].a * 0.2, b=good[-1].b * 0.2,
                                   rho=good[-1].rho, m=good[-1].m,
                                   sigma=good[-1].sigma, T=good[-1].T + 1.0)]
    CHECK(not check_calendar(bad_cal)["arb_free"],
          "the calendar check actually rejects decreasing total variance")

    # Interpolation must land between the bracketing slices.
    s0, s1 = good[1], good[2]
    Tm = 0.5 * (s0.T + s1.T)
    F = float(ve.forward(surf.spot, Tm, surf.r, surf.q))
    ivm = surf.implied_vol(F, Tm)
    CHECK(np.isfinite(ivm) and 0.01 < ivm < 2.0, "interpolated IV is sane")
    w_interp = ivm ** 2 * Tm
    CHECK(float(s0.total_variance(0.0)) - 1e-9 <= w_interp
          <= float(s1.total_variance(0.0)) + 1e-9,
          "interpolated total variance lies between the two bracketing slices")


# ===========================================================================
def main() -> None:
    t0 = time.perf_counter()
    print("=" * 74)
    print("  volengine validation suite")
    print("=" * 74)
    test_black_scholes()
    test_greeks()
    test_implied_vol()
    test_monte_carlo()
    test_garch()
    test_surface()
    dt = time.perf_counter() - t0
    print("\n" + "=" * 74)
    print(f"  {'PASSED' if FAILS == 0 else 'FAILED'}   "
          f"{CHECKS} checks, {FAILS} failures, {dt:.1f}s")
    print("=" * 74 + "\n")
    sys.exit(0 if FAILS == 0 else 1)


if __name__ == "__main__":
    main()
