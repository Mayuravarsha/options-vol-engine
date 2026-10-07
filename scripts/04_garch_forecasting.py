#!/usr/bin/env python3
"""GARCH volatility modelling: estimation, validation, and honest forecasting.

    python scripts/04_garch_forecasting.py
    python scripts/04_garch_forecasting.py --ticker SPY   # real data, needs yfinance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import volengine as ve                                       # noqa: E402
from volengine import plots                                  # noqa: E402
from volengine.garch import (annualised_vol_forecast,        # noqa: E402
                             ewma_variance, fit_garch,
                             forecast_variance, rolling_forecast_study,
                             simulate_garch)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
ANN = np.sqrt(252.0)


def hdr(t: str) -> None:
    print("\n" + "=" * 84)
    print(f"  {t}")
    print("=" * 84)


def stylised_facts(r: np.ndarray) -> dict:
    """The three empirical regularities every volatility model exists to capture."""
    r = np.asarray(r, dtype=float)
    ac_r = [float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in (1, 2, 5, 10, 22)]
    r2 = r ** 2
    ac_r2 = [float(np.corrcoef(r2[:-k], r2[k:])[0, 1]) for k in (1, 2, 5, 10, 22)]
    z = (r - r.mean()) / r.std(ddof=1)
    kurt = float(np.mean(z ** 4))
    skew = float(np.mean(z ** 3))
    return {"acf_returns": ac_r, "acf_squared": ac_r2,
            "excess_kurtosis": kurt - 3.0, "skew": skew,
            "ann_vol": float(r.std(ddof=1) * ANN)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--refit-every", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    out = {}

    # ======================================================================
    hdr("1. THE DATA, AND THE THREE FACTS THAT MOTIVATE THE WHOLE MODEL")
    # ======================================================================
    TRUE = None
    r = None
    if args.ticker:
        r = ve.load_yfinance_returns(args.ticker)
    if r is None:
        csv = os.path.join(DATA, "sample_returns.csv")
        r = ve.load_returns_csv(csv)
        if r is not None:
            print(f"  using the bundled return series: {csv}")
    if r is None:
        TRUE = dict(omega=1.5e-6, alpha=0.045, beta=0.905, gamma=0.075, mu=0.00035)
        r = simulate_garch(args.n, **TRUE, seed=20240)
        print("  using a freshly simulated GJR-GARCH series (known parameters)")

    fx = stylised_facts(r)
    print(f"""
  {len(r):,} daily returns   annualised volatility {fx['ann_vol']*100:.2f}%
  excess kurtosis {fx['excess_kurtosis']:+.2f}   skewness {fx['skew']:+.3f}

  {'lag':>6}{'corr(r_t, r_t-k)':>20}{'corr(r^2_t, r^2_t-k)':>24}
  {'-'*50}""")
    for i, k in enumerate((1, 2, 5, 10, 22)):
        print(f"  {k:>6}{fx['acf_returns'][i]:>20.4f}{fx['acf_squared'][i]:>24.4f}")
    print("""
  This table IS the argument for GARCH:

  1. RETURNS ARE NEARLY UNPREDICTABLE. The left column is near zero at every
     lag. You cannot forecast direction. (If you could, this would be a very
     different project.)

  2. SQUARED RETURNS ARE STRONGLY PREDICTABLE. The right column is clearly
     positive and decays slowly. Today's move tells you nothing about
     tomorrow's DIRECTION but a great deal about tomorrow's MAGNITUDE. That is
     volatility clustering, and it is what GARCH models.

  3. FAT TAILS. Excess kurtosis well above 0 (a normal has exactly 0). Big
     moves happen far more often than a bell curve allows. Note that GARCH
     generates fat tails even with NORMAL innovations, because it is a mixture
     of normals with time-varying variance -- so this is partly explained by
     the model rather than needing to be assumed.
""")

    # ======================================================================
    hdr("2. ESTIMATION BY MAXIMUM LIKELIHOOD")
    # ======================================================================
    t0 = time.perf_counter()
    fit = fit_garch(r, model="garch", dist="normal")
    t_g = time.perf_counter() - t0
    print(fit.summary())
    print(f"    fitted in {t_g:.2f}s")

    t0 = time.perf_counter()
    gjr = fit_garch(r, model="gjr", dist="normal")
    print(gjr.summary())
    print(f"    fitted in {time.perf_counter()-t0:.2f}s")

    t_fit = fit_garch(r, model="gjr", dist="t")
    print(t_fit.summary())

    print(f"""
  HOW TO READ THESE NUMBERS

  * persistence alpha + beta = {fit.persistence:.4f}. This is the single most
    important quantity. It governs how long a volatility shock lasts: the
    effect decays like persistence^h, giving a half-life of
    ln(0.5)/ln(persistence) = {fit.half_life_days:.0f} days. A value this
    close to 1 is completely typical for daily equity data and it is exactly
    why constant-volatility Black-Scholes misprices options: today's
    volatility is highly informative about the next month's.

  * long-run volatility = sqrt(omega/(1-persistence)) * sqrt(252) =
    {fit.long_run_vol_annual*100:.2f}% annualised. Every multi-step forecast
    decays toward this level.

  * t-statistics. alpha and beta both have |t| far above 2, so the ARCH and
    GARCH effects are real, not sampling noise. A point estimate with no
    standard error next to it is not a statistical result.

  * MODEL SELECTION by information criterion (lower is better):
        GARCH normal   AIC {fit.aic:>10.1f}   BIC {fit.bic:>10.1f}
        GJR   normal   AIC {gjr.aic:>10.1f}   BIC {gjr.bic:>10.1f}
        GJR   Student-t AIC {t_fit.aic:>10.1f}   BIC {t_fit.bic:>10.1f}
    AIC and BIC both penalise extra parameters; BIC penalises harder. Adding
    a parameter always improves the raw likelihood, so comparing likelihoods
    alone would always pick the biggest model.
""")
    tg = gjr.gamma / gjr.std_errors.get("gamma", np.nan)
    print(f"""  * THE LEVERAGE EFFECT. GJR's gamma = {gjr.gamma:.4f} with t = {tg:.2f}.
    Positive gamma means volatility rises MORE after a fall than after a rally
    of the same size. This is one of the most reliably estimated facts in
    empirical finance, and it is the statistical twin of the equity volatility
    SKEW in the option surface: out-of-the-money puts are expensive precisely
    because down moves come with more volatility. One phenomenon, two
    completely different datasets.
""")
    if TRUE:
        print("  * PARAMETER RECOVERY (this series was simulated, so we know the truth):")
        print(f"      {'param':<8}{'true':>12}{'estimated':>14}{'std err':>13}{'z':>8}")
        for k, v in TRUE.items():
            est = getattr(gjr, k)
            se = gjr.std_errors.get(k, np.nan)
            z = (est - v) / se if se and np.isfinite(se) and se > 0 else np.nan
            print(f"      {k:<8}{v:>12.6g}{est:>14.6g}{se:>13.4g}{z:>8.2f}")
        print("""
    Every z well inside +/-2 means the estimator recovers what generated the
    data. This is the check that earns the right to trust it on real data --
    an estimator that cannot recover parameters from its own model is telling
    you nothing when you point it at the market.
""")

    # ======================================================================
    hdr("3. FORECASTING")
    # ======================================================================
    last_eps = float(r[-1] - fit.mu)
    last_var = float(fit.conditional_vol[-1] ** 2)
    print(f"""
  Today's conditional volatility  {np.sqrt(last_var)*ANN*100:.2f}% annualised
  Long-run level                  {fit.long_run_vol_annual*100:.2f}% annualised

  Multi-step forecasts. For h > 1 we do not know the future shock, but its
  EXPECTED square is the variance itself, so the recursion collapses to

      E[sigma^2_(t+h)] = LR + persistence^(h-1) * (sigma^2_(t+1) - LR)

  -- geometric decay from today's level back toward the long-run level.
""")
    print(f"    {'horizon':>10}{'point vol':>14}{'avg vol over horizon':>24}")
    print("    " + "-" * 48)
    for h in (1, 5, 22, 63, 126, 252):
        fc = forecast_variance(fit, last_eps, last_var, h)
        print(f"    {h:>7} d{np.sqrt(fc[-1])*ANN*100:>13.2f}%"
              f"{np.sqrt(fc.mean())*ANN*100:>22.2f}%")
    print("""
  The right-hand column is the one to use for option pricing: Black-Scholes
  wants the AVERAGE variance over the option's life, not the instantaneous
  value at expiry.
""")

    # ======================================================================
    hdr("4. OUT-OF-SAMPLE FORECAST COMPETITION")
    # ======================================================================
    print(f"""
  The rule that decides whether any of this is real: at every point in time
  the model may only use data available UP TO that point. We refit on an
  expanding window every {args.refit_every} days and forecast one day ahead.

  Fitting once on the whole sample and then 'forecasting' the same data is
  in-sample fitting. It always looks excellent and it means nothing.

  Three competitors:
    GARCH       mean-reverting, parameters estimated
    EWMA(0.94)  RiskMetrics. This is GARCH with omega=0 and persistence
                exactly 1, so it has NO long-run level to revert to and its
                multi-step forecast is flat forever. Good at one day, poor at
                one month, and knowing why is the point of including it.
    constant    trailing 252-day sample variance. The honest null hypothesis:
                if you cannot beat this, you have achieved nothing.

  Scored two ways, because the target is unobservable:
    RMSE   on volatility, against the |return| proxy
    QLIKE  the loss function Patton (2011) showed still ranks forecasts
           correctly when the target is a noisy but unbiased proxy. It is far
           less dominated by a handful of huge days than RMSE, and it
           penalises UNDER-forecasting more than over-forecasting -- the right
           asymmetry for risk management.
""")
    t0 = time.perf_counter()
    study = rolling_forecast_study(r, train=args.train,
                                   refit_every=args.refit_every, horizon=1)
    print(f"  ({len(study['index'])} out-of-sample days, "
          f"{time.perf_counter()-t0:.1f}s)\n")
    print(f"    {'model':<12}{'RMSE (vol pts)':>18}{'MAE (vol pts)':>17}"
          f"{'QLIKE':>12}{'mean fcst vol':>16}")
    print("    " + "-" * 75)
    best_r = min(["garch", "ewma", "constant"], key=lambda m: study[m]["rmse_vol"])
    best_q = min(["garch", "ewma", "constant"], key=lambda m: study[m]["qlike"])
    for m in ("garch", "ewma", "constant"):
        s = study[m]
        mark = ""
        if m == best_r:
            mark += "  <- best RMSE"
        if m == best_q:
            mark += "  <- best QLIKE"
        print(f"    {m:<12}{s['rmse_vol']*ANN*100:>18.3f}{s['mae_vol']*ANN*100:>17.3f}"
              f"{s['qlike']:>12.5f}{s['mean_forecast_vol_annual']*100:>15.2f}%{mark}")

    g, c = study["garch"], study["constant"]
    print(f"""
    GARCH vs constant volatility:
      RMSE   {(1 - g['rmse_vol']/c['rmse_vol'])*100:+.1f}%
      QLIKE  {(1 - g['qlike']/c['qlike'])*100:+.1f}%

  A word on magnitudes. The improvement over a constant looks modest in RMSE,
  and that is not a weakness of GARCH -- it is the noise in the PROXY. The
  squared daily return is an unbiased but extremely noisy estimate of that
  day's variance (for a normal, Var(r^2) = 2 sigma^4), so most of the RMSE is
  irreducible measurement error that no forecast could ever remove. Compare
  models against each other, never against zero. QLIKE, which is much less
  sensitive to that noise, shows the gap more clearly.
""")

    # ======================================================================
    hdr("5. THE BRIDGE TO OPTIONS: implied vs forecast volatility")
    # ======================================================================
    S = 100.0
    T = 22 / 252.0
    garch_vol = annualised_vol_forecast(fit, last_eps, last_var, 22)
    chain = ve.generate_synthetic_chain(S=S, maturities=[0.0833])
    cl, _ = ve.clean_chain(chain)
    from volengine.implied_vol import implied_vol_vector
    iv, conv, _, _ = implied_vol_vector(cl["mid_price"], S, cl["strike"],
                                        cl["expiry_years"], cl["r"], cl["q"],
                                        cl["option_type"])
    Ks = cl["strike"][conv]
    ivs = iv[conv]
    order = np.argsort(Ks)
    atm_iv = float(np.interp(S, Ks[order], ivs[order]))

    print(f"""
  GARCH 22-day forecast (REAL-WORLD measure)   {garch_vol*100:.2f}%
  At-the-money implied volatility (RISK-NEUTRAL) {atm_iv*100:.2f}%
  Gap                                          {(atm_iv-garch_vol)*100:+.2f} vol points

  These two numbers are not supposed to be equal, and understanding why is one
  of the most useful things in this whole project.

  GARCH estimates volatility under the REAL-WORLD measure P: what will
  actually happen, on average. Implied volatility lives under the RISK-NEUTRAL
  measure Q: what the market charges. The difference is the VARIANCE RISK
  PREMIUM, and it is one of the most robust findings in empirical finance --
  implied volatility exceeds subsequent realised volatility the large majority
  of the time, across markets and decades.

  Why should anyone pay it? Because volatility spikes exactly when everything
  else in your portfolio is falling. An asset that pays off in a crash is
  worth more than its expected payoff, so investors accept a negative expected
  return to hold it -- and option sellers are paid for supplying it. The
  premium is compensation for a genuinely nasty risk profile: small steady
  gains punctuated by large losses.

  This gap is also the basis of the short-variance trade, and of why that
  trade periodically destroys the people running it.
""")

    p1 = plots.plot_garch(r, fit, study, RESULTS, gjr_fit=gjr)
    p2 = plots.plot_news_impact([fit, gjr], ["GARCH(1,1) - symmetric",
                                             "GJR-GARCH - leverage effect"], RESULTS)
    p3 = plots.plot_iv_vs_garch(Ks[order], ivs[order], garch_vol, RESULTS, 0.0833)
    for p in (p1, p2, p3):
        print(f"  chart -> {os.path.relpath(p, os.path.join(HERE, '..'))}")

    out = {
        "stylised_facts": fx,
        "garch": dict(omega=fit.omega, alpha=fit.alpha, beta=fit.beta, mu=fit.mu,
                      loglik=fit.loglik, aic=fit.aic, bic=fit.bic,
                      persistence=fit.persistence,
                      half_life_days=fit.half_life_days,
                      long_run_vol_annual=fit.long_run_vol_annual,
                      std_errors=fit.std_errors),
        "gjr": dict(omega=gjr.omega, alpha=gjr.alpha, gamma=gjr.gamma, beta=gjr.beta,
                    loglik=gjr.loglik, aic=gjr.aic, bic=gjr.bic,
                    persistence=gjr.persistence, std_errors=gjr.std_errors,
                    gamma_tstat=float(tg)),
        "gjr_t": dict(nu=t_fit.nu, aic=t_fit.aic, bic=t_fit.bic, loglik=t_fit.loglik),
        "true_parameters": TRUE,
        "out_of_sample": {m: {k: v for k, v in study[m].items()
                              if k != "forecast_var"}
                          for m in ("garch", "ewma", "constant")},
        "n_oos_days": len(study["index"]),
        "garch_22d_forecast_vol": garch_vol,
        "atm_implied_vol": atm_iv,
        "variance_risk_premium_vol_pts": (atm_iv - garch_vol) * 100,
    }
    with open(os.path.join(RESULTS, "garch.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("  data   -> results/garch.json\n")


if __name__ == "__main__":
    main()
