"""Figures for the volatility engine."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401,E402

INK = "#1f2933"
ACCENT = "#2563eb"
WARM = "#c2410c"
GREEN = "#15803d"
PURPLE = "#7c3aed"
GREY = "#94a3b8"
SERIES = [ACCENT, WARM, GREEN, PURPLE, "#0891b2", "#be123c", "#a16207"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def plot_greeks(S_grid, greeks_by_T: Dict[float, Dict], outdir: str,
                K: float = 100.0, name: str = "greeks.png") -> str:
    """Each Greek against spot, one line per maturity."""
    keys = ["price", "delta", "gamma", "vega", "theta", "rho"]
    titles = {
        "price": "Price",
        "delta": "Delta  dV/dS  (also the hedge ratio)",
        "gamma": "Gamma  d2V/dS2  (peaks ATM, explodes near expiry)",
        "vega": "Vega  dV/dsigma  (largest for LONG-dated ATM)",
        "theta": "Theta  dV/dt per year  (decay accelerates near expiry)",
        "rho": "Rho  dV/dr",
    }
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax, key in zip(axs.ravel(), keys):
        for i, (T, g) in enumerate(sorted(greeks_by_T.items())):
            ax.plot(S_grid, g[key], color=SERIES[i % len(SERIES)], lw=1.4,
                    label=f"T = {T:g}y")
        ax.axvline(K, color=GREY, ls="--", lw=0.9)
        ax.set_title(titles[key], loc="left", fontsize=9)
        ax.set_xlabel("spot")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Black-Scholes Greeks, call, K = {K:g}, vertical line = strike",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, outdir, name)


def plot_iv_convergence(results: Dict, outdir: str,
                        name: str = "iv_convergence.png") -> str:
    """Newton's convergence path and where it needs the Brent fallback."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))

    for i, (label, path) in enumerate(results["paths"].items()):
        axs[0].semilogy(range(len(path)), np.maximum(np.abs(path), 1e-17),
                        "o-", color=SERIES[i % len(SERIES)], lw=1.3, label=label, ms=3)
    axs[0].set_xlabel("Newton iteration")
    axs[0].set_ylabel("|price error| (log scale)")
    axs[0].set_title("Quadratic convergence:\ndigits double each step", loc="left")
    axs[0].legend(frameon=False, fontsize=8)

    m = results["moneyness"]
    axs[1].plot(m, results["vega"], color=ACCENT, lw=1.4)
    axs[1].set_xlabel("K / S")
    axs[1].set_ylabel("vega")
    axs[1].set_title("Vega collapses in the wings --\nthis is why Newton fails there",
                     loc="left")

    axs[2].plot(m, results["iterations"], "o-", color=GREEN, ms=3, lw=1.0)
    fails = np.asarray(results["failed_moneyness"])
    if fails.size:
        axs[2].plot(fails, np.zeros_like(fails), "x", color=WARM, ms=8,
                    label="Newton failed -> Brent")
        axs[2].legend(frameon=False, fontsize=8)
    axs[2].set_xlabel("K / S")
    axs[2].set_ylabel("iterations to converge")
    axs[2].set_title("Cost of inversion across the chain", loc="left")
    return _save(fig, outdir, name)


def plot_surface(surf, chain, outdir: str, name: str = "vol_surface.png",
                 true_iv_fn=None) -> str:
    """Smile slices (left) and the 3D surface (right)."""
    from .black_scholes import forward

    fig = plt.figure(figsize=(13.5, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    slices = sorted(surf.slices, key=lambda s: s.T)

    for i, s in enumerate(slices):
        col = SERIES[i % len(SERIES)]
        F = float(forward(surf.spot, s.T, surf.r, surf.q))
        sel = chain["expiry_years"] == s.T
        if np.any(sel):
            kk = np.log(chain["strike"][sel] / F)
            from .implied_vol import implied_vol_vector
            iv, conv, _, _ = implied_vol_vector(
                chain["mid_price"][sel], surf.spot, chain["strike"][sel],
                s.T, surf.r, surf.q, chain["option_type"][sel])
            ax1.plot(kk[conv], iv[conv] * 100, "o", color=col, ms=3.2, alpha=0.6)
        # Plot the fit only a little past the strikes it was actually fit to.
        # SVI happily extrapolates to absurd values outside that window (a
        # steep short-dated slice can read 300% a few tenths of k past its
        # last quote) -- showing that is not "the fit", it is an artifact of
        # asking a 5-parameter curve about a region it never saw any data
        # from. Fall back to a fixed, generous window when k_lo/k_hi were
        # never set (e.g. a VolSurface built by code that predates them).
        if np.isfinite(s.k_lo) and np.isfinite(s.k_hi):
            kg = np.linspace(s.k_lo - 0.15, s.k_hi + 0.15, 300)
        else:
            kg = np.linspace(-0.9, 0.55, 300)
        ax1.plot(kg, s.implied_vol(kg) * 100, color=col, lw=1.5,
                 label=f"T={s.T:.3g}y  (fit RMSE {s.rmse_vol*100:.2f})")
        if true_iv_fn is not None:
            try:
                Kg = F * np.exp(kg)
                ax1.plot(kg, true_iv_fn(Kg, s.T) * 100, color=col, lw=0.9, ls=":")
            except Exception:
                pass

    ax1.axvline(0, color=GREY, ls="--", lw=0.9)
    ax1.set_xlabel("log-moneyness  k = log(K / F)")
    ax1.set_ylabel("implied volatility (%)")
    ax1.set_title("Volatility smile by expiry\ndots = market quotes, "
                  "solid = SVI fit, dotted = truth", loc="left")
    ax1.legend(frameon=False, fontsize=7.5)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    Ts = np.array([s.T for s in slices])
    # This is a single continuous MESH shared by every maturity, so unlike
    # the left panel (one k-window per slice) there is no way to give a
    # narrow short-dated slice a narrower domain than a long-dated one on the
    # same plot. That forces a choice: use the INTERSECTION of what every
    # slice observed (build_surface already computes exactly this for the
    # calendar check, for the identical reason -- comparing slices at a
    # shared k needs a k both of them actually saw), trading a smaller-
    # looking surface for one with no extrapolation spikes at the corners. A
    # union looks more complete but is not: on this SPY snapshot it produced
    # a spurious ~600% IV spike from evaluating a poorly-covered slice miles
    # outside the strikes that fit it.
    k_range = surf.calendar.get("k_range_checked") if surf.calendar else None
    kg = np.linspace(*k_range, 60) if k_range else np.linspace(-0.7, 0.45, 60)
    Tg = np.linspace(Ts.min(), Ts.max(), 60)
    KK, TT = np.meshgrid(kg, Tg)
    ZZ = np.zeros_like(KK)
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            F = float(forward(surf.spot, TT[i, j], surf.r, surf.q))
            ZZ[i, j] = surf.implied_vol(F * np.exp(KK[i, j]), TT[i, j]) * 100
    ax2.plot_surface(KK, TT, ZZ, cmap="viridis", alpha=0.92,
                     linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax2.set_xlabel("log-moneyness")
    ax2.set_ylabel("maturity (years)")
    ax2.set_zlabel("IV (%)")
    ax2.set_title("Implied volatility surface", loc="left")
    ax2.view_init(elev=24, azim=-125)
    return _save(fig, outdir, name)


def plot_smiles_grid(surf, chain, outdir: str, name: str = "smiles_grid.png",
                     true_iv_fn=None, ncols: int = 4, title: str = "") -> str:
    """One subplot PER EXPIRY: market quotes (dots) against the SVI fit
    (line), each on its own axes.

    Why this exists alongside plot_surface's left panel: that panel overlays
    every expiry on ONE set of axes, which is the right view for comparing
    the term structure's overall shape but becomes unreadable past three or
    four expiries -- eight overlapping scatter series in different colours is
    noise, not a chart. This is the complementary view: worse for comparing
    expiries against each other, far better for judging any ONE expiry's fit
    quality (are the dots actually near the line? where does it miss?) in
    isolation, which is the question you ask when a specific slice looks
    wrong in the summary table.
    """
    from .black_scholes import forward
    from .implied_vol import implied_vol_vector

    slices = sorted(surf.slices, key=lambda s: s.T)
    n = len(slices)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.1 * nrows), squeeze=False)

    for i, s in enumerate(slices):
        ax = axs[i // ncols][i % ncols]
        col = SERIES[i % len(SERIES)]
        F = float(forward(surf.spot, s.T, surf.r, surf.q))
        sel = chain["expiry_years"] == s.T
        if np.any(sel):
            kk = np.log(chain["strike"][sel] / F)
            iv, conv, _, _ = implied_vol_vector(
                chain["mid_price"][sel], surf.spot, chain["strike"][sel],
                s.T, surf.r, surf.q, chain["option_type"][sel])
            ax.plot(kk[conv], iv[conv] * 100, "o", color=col, ms=3.0, alpha=0.55,
                    label="market" if i == 0 else None)
        lo = s.k_lo if np.isfinite(s.k_lo) else -0.6
        hi = s.k_hi if np.isfinite(s.k_hi) else 0.6
        kg = np.linspace(lo - 0.15, hi + 0.15, 250)
        ax.plot(kg, s.implied_vol(kg) * 100, color=col, lw=1.6,
                label="SVI fit" if i == 0 else None)
        if true_iv_fn is not None:
            try:
                Kg = F * np.exp(kg)
                ax.plot(kg, true_iv_fn(Kg, s.T) * 100, color=GREY, lw=0.9, ls=":",
                        label="truth" if i == 0 else None)
            except Exception:
                pass
        ax.axvline(0, color=GREY, ls="--", lw=0.7)
        trust = "" if s.reliable() else "  [extrapolated]"
        ax.set_title(f"T={s.T*365:.0f}d   n={s.n_points}   "
                     f"RMSE={s.rmse_vol*100:.2f}vp{trust}", fontsize=8, loc="left")
        if i // ncols == nrows - 1:
            ax.set_xlabel("log-moneyness k")
        if i % ncols == 0:
            ax.set_ylabel("IV (%)")

    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    axs[0][0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(f"{title + '  ' if title else ''}Volatility smile by expiry "
                 "(one panel per maturity)", x=0.01, ha="left", fontsize=11)
    return _save(fig, outdir, name)


def plot_term_structure(curves: Dict[str, Sequence], outdir: str,
                        name: str = "term_structure.png",
                        skew_curves: Optional[Dict[str, Sequence]] = None) -> str:
    """ATM implied vol against maturity -- the TERM STRUCTURE -- with an
    optional second panel for a skew metric against maturity.

    `curves` is {label: [(T_years, atm_vol), ...]}, plain numbers rather than
    VolSurface objects on purpose: this plot is meant to be produced from the
    small JSON summary a calibration run already writes to results/, so that
    comparing SPY against AAPL against TSLA does not require re-fitting a
    surface (each fit costs real wall-clock time) just to draw one line per
    ticker.

    `skew_curves`, if given, is {label: [(T_years, skew), ...]} where `skew`
    should already be a well-defined, MODEL-ROBUST number (e.g. IV(25-delta
    put) - IV(25-delta call), or the fixed-moneyness difference this project
    uses -- see docs/CONCEPTS.md's note on why raw SVI rho is NOT used for
    this: rho is only a reliable skew indicator when the fitted vertex m
    falls inside the strikes actually observed, which real OTM-only chains do
    not guarantee, especially for short-dated slices).
    """
    ncols = 2 if skew_curves else 1
    fig, axs = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.2))
    axs = np.atleast_1d(axs)

    for i, (label, pts) in enumerate(curves.items()):
        pts = sorted(pts, key=lambda p: p[0])
        Ts = np.array([p[0] for p in pts]) * 365.0
        vols = np.array([p[1] for p in pts]) * 100.0
        axs[0].plot(Ts, vols, "o-", color=SERIES[i % len(SERIES)], lw=1.6, ms=4.5, label=label)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("days to expiry (log scale)")
    axs[0].set_ylabel("ATM implied volatility (%)")
    axs[0].set_title("Term structure: ATM vol vs maturity", loc="left")
    axs[0].legend(frameon=False, fontsize=8)

    if skew_curves:
        for i, (label, pts) in enumerate(skew_curves.items()):
            pts = sorted(pts, key=lambda p: p[0])
            Ts = np.array([p[0] for p in pts]) * 365.0
            sk = np.array([p[1] for p in pts]) * 100.0
            axs[1].plot(Ts, sk, "o-", color=SERIES[i % len(SERIES)], lw=1.6, ms=4.5, label=label)
        axs[1].axhline(0, color=GREY, ls="--", lw=0.9)
        axs[1].set_xscale("log")
        axs[1].set_xlabel("days to expiry (log scale)")
        axs[1].set_ylabel("skew: IV(k=-0.10) - IV(k=+0.10)  (vol points)")
        axs[1].set_title("Skew term structure\n"
                         "positive = puts richer than calls = crash-risk premium", loc="left")
        axs[1].legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def plot_arbitrage_checks(surf, outdir: str, name: str = "arbitrage_checks.png") -> str:
    """Durrleman's g (butterfly) and total variance monotonicity (calendar)."""
    from .surface import durrleman_g
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.7))
    k = np.linspace(-1.2, 0.8, 500)
    slices = sorted(surf.slices, key=lambda s: s.T)

    for i, s in enumerate(slices):
        col = SERIES[i % len(SERIES)]
        axs[0].plot(k, durrleman_g(s, k), color=col, lw=1.3, label=f"T={s.T:.3g}y")
        axs[1].plot(k, s.total_variance(k), color=col, lw=1.3, label=f"T={s.T:.3g}y")
    axs[0].axhline(0, color=WARM, ls="--", lw=1.2)
    axs[0].set_xlabel("log-moneyness")
    axs[0].set_ylabel("g(k)")
    axs[0].set_title("BUTTERFLY: Durrleman's g must stay above zero\n"
                     "(below zero = negative implied probability)", loc="left")
    axs[0].legend(frameon=False, fontsize=7.5)

    axs[1].set_xlabel("log-moneyness")
    axs[1].set_ylabel("total implied variance  w = sigma^2 T")
    axs[1].set_title("CALENDAR: curves must never cross\n"
                     "(variance cannot un-accumulate)", loc="left")
    axs[1].legend(frameon=False, fontsize=7.5)
    return _save(fig, outdir, name)


def plot_mc_convergence(rows: Dict, outdir: str,
                        name: str = "mc_convergence.png") -> str:
    n = np.asarray(rows["n"], dtype=float)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    labels = [k for k in rows if k not in ("n", "exact")]

    for i, lab in enumerate(labels):
        col = SERIES[i % len(SERIES)]
        axs[0].loglog(n, rows[lab]["stderr"], "o-", color=col, lw=1.3, ms=3.5, label=lab)
        axs[1].loglog(n, np.maximum(rows[lab]["abs_error"], 1e-8), "o-",
                      color=col, lw=1.3, ms=3.5, label=lab)
    ref = rows[labels[0]]["stderr"][0] * np.sqrt(n[0] / n)
    axs[0].loglog(n, ref, "--", color=GREY, lw=1.1, label="1/sqrt(N) reference")
    axs[0].set_xlabel("paths N")
    axs[0].set_ylabel("standard error")
    axs[0].set_title("Standard error falls like 1/sqrt(N)\n"
                     "variance reduction shifts the line down, not its slope", loc="left")
    axs[0].legend(frameon=False, fontsize=7.5)

    axs[1].set_xlabel("paths N")
    axs[1].set_ylabel("|MC price - Black-Scholes|")
    axs[1].set_title("Actual error vs the exact answer", loc="left")

    base = np.asarray(rows[labels[0]]["variance"], dtype=float)
    for i, lab in enumerate(labels[1:], start=1):
        axs[2].semilogx(n, base / np.asarray(rows[lab]["variance"]),
                        "o-", color=SERIES[i % len(SERIES)], lw=1.3, ms=3.5, label=lab)
    axs[2].axhline(1, color=GREY, ls="--", lw=1.0)
    axs[2].set_xlabel("paths N")
    axs[2].set_ylabel("variance reduction factor (x)")
    axs[2].set_title("How many times fewer paths you need\nfor the same accuracy", loc="left")
    axs[2].legend(frameon=False, fontsize=7.5)
    return _save(fig, outdir, name)


def plot_garch(returns, fit, forecast_study: Optional[Dict], outdir: str,
               gjr_fit=None, name: str = "garch.png") -> str:
    r = np.asarray(returns)
    ann = np.sqrt(252) * 100
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.2))

    axs[0].plot(r * 100, color=GREY, lw=0.5)
    axs[0].set_ylabel("daily return (%)")
    axs[0].set_title("Returns: note the clusters -- calm stretches and violent "
                     "stretches, not a constant", loc="left")

    cv = np.asarray(fit.conditional_vol) * ann
    axs[1].plot(cv, color=ACCENT, lw=1.0, label=f"{fit.model.upper()} conditional vol")
    if gjr_fit is not None:
        axs[1].plot(np.asarray(gjr_fit.conditional_vol) * ann, color=WARM, lw=1.0,
                    alpha=0.8, label="GJR conditional vol")
    axs[1].axhline(fit.long_run_vol_annual * 100, color=GREEN, ls="--", lw=1.1,
                   label=f"long-run {fit.long_run_vol_annual*100:.1f}%")
    axs[1].set_ylabel("annualised vol (%)")
    axs[1].set_title(f"Conditional volatility  |  persistence "
                     f"{fit.persistence:.4f}, shock half-life "
                     f"{fit.half_life_days:.0f} days", loc="left")
    axs[1].legend(frameon=False, fontsize=8)

    if forecast_study:
        idx = np.asarray(forecast_study["index"])
        act = np.sqrt(np.asarray(forecast_study["actual_var_proxy"])) * ann
        axs[2].plot(idx, act, color=GREY, lw=0.5, alpha=0.75, label="|return| (noisy proxy)")
        for i, m in enumerate(["garch", "ewma", "constant"]):
            axs[2].plot(idx, np.sqrt(np.asarray(forecast_study[m]["forecast_var"])) * ann,
                        color=SERIES[i], lw=1.1,
                        label=f"{m}  RMSE(vol) {forecast_study[m]['rmse_vol']*ann:.2f}  "
                              f"QLIKE {forecast_study[m]['qlike']:.4f}")
        axs[2].set_ylabel("annualised vol (%)")
        axs[2].set_xlabel("day index")
        axs[2].set_title("OUT-OF-SAMPLE one-day-ahead forecasts (expanding window, "
                         "periodic refit)", loc="left")
        axs[2].legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def plot_news_impact(fits: Sequence, labels: Sequence[str], outdir: str,
                     name: str = "news_impact_curve.png") -> str:
    """Engle-Ng news impact curve: today's variance vs yesterday's shock."""
    from .garch import news_impact_curve
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (f, lab) in enumerate(zip(fits, labels)):
        eps, var = news_impact_curve(f)
        ax.plot(eps * 100, np.sqrt(var) * np.sqrt(252) * 100,
                color=SERIES[i % len(SERIES)], lw=1.5, label=lab)
    ax.axvline(0, color=GREY, ls="--", lw=0.9)
    ax.set_xlabel("yesterday's shock (%)")
    ax.set_ylabel("today's annualised vol (%)")
    ax.set_title("News impact curve: GARCH is symmetric, GJR is not.\n"
                 "The steeper LEFT arm is the leverage effect.", loc="left")
    ax.legend(frameon=False)
    return _save(fig, outdir, name)


def plot_iv_vs_garch(strikes, market_iv, garch_vol, outdir: str, T: float,
                     name: str = "iv_vs_garch.png") -> str:
    """Implied vol against the GARCH real-world forecast: the risk premium."""
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot(strikes, np.asarray(market_iv) * 100, "o-", color=ACCENT, ms=3.5,
            lw=1.3, label="implied volatility (risk-neutral)")
    ax.axhline(garch_vol * 100, color=WARM, ls="--", lw=1.4,
               label=f"GARCH forecast (real-world) {garch_vol*100:.1f}%")
    ax.fill_between(strikes, garch_vol * 100, np.asarray(market_iv) * 100,
                    color=GREEN, alpha=0.14, label="variance risk premium")
    ax.set_xlabel("strike")
    ax.set_ylabel("volatility (%)")
    ax.set_title(f"Implied vs forecast volatility, T = {T:.3g}y\n"
                 "the gap is what option sellers are paid for", loc="left")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)
