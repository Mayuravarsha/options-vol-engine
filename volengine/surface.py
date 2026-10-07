"""The implied volatility surface: SVI fitting with no-arbitrage checks.

=============================================================================
THE SMILE: BLACK-SCHOLES' MOST FAMOUS FAILURE
=============================================================================
Black-Scholes says volatility is a property of the STOCK. If that were true,
every option on the same stock with the same expiry would imply the same
volatility, and a plot of implied vol against strike would be a flat line.

It is not flat. For equity indices it slopes steeply down: low strikes (puts)
imply much higher volatility than high strikes (calls). That shape is called
the SKEW or the smirk. For FX it is closer to a symmetric smile.

The equity skew was mild before October 1987 and has been steep ever since.
Two non-exclusive explanations:

  * CRASH RISK. Real returns have a fat LEFT tail. Black-Scholes assumes a
    lognormal, which does not. The market prices the extra tail risk by paying
    up for out-of-the-money puts, and when you invert that price through a
    model that cannot represent fat tails, the excess has to come out somewhere
    -- it comes out as a higher implied volatility.

  * LEVERAGE / VOLATILITY FEEDBACK. When equity falls, leverage rises and
    volatility rises with it. Down moves and high volatility arrive together.
    This is the same effect the GJR gamma measures in garch.py, seen from the
    options side.

So the implied vol surface is not an embarrassment to be explained away. It is
a rich, tradeable object: it is the market's entire risk-neutral distribution,
written in Black-Scholes units.

=============================================================================
WHY YOU NEED A PARAMETRIC MODEL AND CANNOT JUST INTERPOLATE
=============================================================================
Real chains are sparse, noisy and unevenly spaced. Spline through the points
naively and you get a surface that permits ARBITRAGE -- a set of prices where
somebody can construct a portfolio with a guaranteed profit. Then every price
you interpolate off it is wrong, and any model you calibrate to it inherits the
inconsistency.

SVI (Gatheral, 2004) is the industry standard because it is flexible enough to
fit real smiles, has only five parameters, and its arbitrage conditions are
checkable in closed form.

=============================================================================
SVI, IN RAW PARAMETERISATION
=============================================================================
Work in TOTAL IMPLIED VARIANCE  w(k) = sigma_implied(k)^2 * T,  against
log-moneyness  k = log(K / F)  where F is the forward.

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

Five parameters, each with a clear geometric meaning:

    a      overall level -- lifts the whole smile up or down
    b      slope of the wings; b >= 0. Larger b = steeper smile.
    rho    asymmetry, -1 < rho < 1. Negative rho tilts the smile down to the
           right, which is the equity skew.
    m      horizontal shift -- where the minimum sits in log-moneyness
    sigma  curvature at the bottom; sigma > 0. Small sigma = sharp V,
           large sigma = smooth bowl.

Why TOTAL variance rather than volatility? Because total variance is the thing
that must be monotone in time for no calendar arbitrage, and because the SVI
wings are asymptotically LINEAR in k in total-variance space -- which is
exactly what Roger Lee's moment formula says any arbitrage-free smile must do.
The parameterisation is built to respect that automatically.

=============================================================================
THE TWO ARBITRAGE CONDITIONS -- both are checked below
=============================================================================
BUTTERFLY (static, within one expiry). A butterfly spread -- buy K-dK, sell
two K, buy K+dK -- has a payoff that is never negative, so it must never cost
less than nothing. Equivalently, the implied risk-neutral density must be
non-negative everywhere. Durrleman's condition expresses this directly on the
smile:

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  >=  0

If g(k) < 0 anywhere, your surface implies a NEGATIVE PROBABILITY at that
strike. Everything downstream is then meaningless.

CALENDAR (across expiries). Total implied variance must be non-decreasing in
maturity at every fixed log-moneyness:

    w(k, T1) <= w(k, T2)   for   T1 < T2

Otherwise a calendar spread is free money: variance cannot un-accumulate.

Both are checked and reported rather than silently assumed. A surface that
fails them is not a surface, it is a plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .black_scholes import bs_price, forward
from .implied_vol import implied_vol_vector


# ---------------------------------------------------------------------------
@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    rmse_vol: float = float("nan")
    n_points: int = 0
    converged: bool = False
    n_outliers_dropped: int = 0
    # The log-moneyness range actually COVERED by the strikes this slice was
    # fit to. SVI has 5 free parameters; if the observed strikes only span a
    # narrow or one-sided window (routine on a real, OTM-only chain -- e.g. a
    # short-dated slice where deep discounts get filtered out entirely), the
    # fit can match that narrow window closely while the curve's shape
    # everywhere else is pure extrapolation with no data behind it. `k_lo`,
    # `k_hi` record that window so `reliable()` below can say so honestly.
    k_lo: float = float("nan")
    k_hi: float = float("nan")

    def reliable(self, extrap_tol: float = 0.3) -> bool:
        """False when the fitted vertex (m -- where the smile's curvature is
        centred) lies outside the strikes this slice actually observed, plus
        a small tolerance. When that happens, parameters like rho -- whose
        TEXTBOOK reading is "the sign of the skew" -- stop meaning that: the
        optimiser is free to place the vertex whichever side is convenient
        for reproducing the visible data. It is not a wrong answer to the fit
        problem it was given; it is the wrong question to read the sign of
        rho from. See docs/CONCEPTS.md for the real slice that motivated this
        (SPY, 13 days to expiry: rho fit at +0.997, k range only [-0.25,
        +0.05] -- an obviously equity-skewed smile whose SVI parameters read
        as if it were skewed the other way)."""
        if not (np.isfinite(self.k_lo) and np.isfinite(self.k_hi)):
            return True
        return (self.k_lo - extrap_tol) <= self.m <= (self.k_hi + extrap_tol)

    def total_variance(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k) -> np.ndarray:
        w = np.maximum(self.total_variance(k), 1e-12)
        return np.sqrt(w / self.T)

    def derivatives(self, k):
        """w, w', w'' -- needed for the butterfly check."""
        k = np.asarray(k, dtype=float)
        d = k - self.m
        root = np.sqrt(d ** 2 + self.sigma ** 2)
        w = self.a + self.b * (self.rho * d + root)
        w1 = self.b * (self.rho + d / root)
        w2 = self.b * self.sigma ** 2 / root ** 3
        return w, w1, w2

    def __repr__(self) -> str:
        return (f"SVI(T={self.T:.4f}, a={self.a:.5f}, b={self.b:.5f}, "
                f"rho={self.rho:+.4f}, m={self.m:+.5f}, sigma={self.sigma:.5f}, "
                f"rmse={self.rmse_vol*100:.3f} vol pts, n={self.n_points})")


def durrleman_g(params: SVIParams, k) -> np.ndarray:
    """Durrleman's function. Negative anywhere => butterfly arbitrage."""
    w, w1, w2 = params.derivatives(k)
    w = np.maximum(w, 1e-12)
    return (1.0 - np.asarray(k) * w1 / (2.0 * w)) ** 2 \
        - (w1 ** 2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


def check_butterfly(params: SVIParams, k_range: Tuple[float, float] = (-1.5, 1.5),
                    n: int = 1000) -> Dict[str, float]:
    k = np.linspace(*k_range, n)
    g = durrleman_g(params, k)
    worst = int(np.argmin(g))
    return {"min_g": float(g[worst]), "arb_free": bool(g.min() >= -1e-10),
            "worst_k": float(k[worst]),
            "pct_violating": float(100.0 * np.mean(g < 0))}


def check_calendar(slices: Sequence[SVIParams],
                   k_range: Tuple[float, float] = (-1.0, 1.0),
                   n: int = 400) -> Dict[str, float]:
    """Total variance must not decrease with maturity at any log-moneyness."""
    s = sorted(slices, key=lambda p: p.T)
    if len(s) < 2:
        return {"arb_free": True, "worst_violation": 0.0, "n_pairs": 0}
    k = np.linspace(*k_range, n)
    worst = 0.0
    viol = 0
    for i in range(len(s) - 1):
        w1 = s[i].total_variance(k)
        w2 = s[i + 1].total_variance(k)
        d = w2 - w1
        if d.min() < worst:
            worst = float(d.min())
        viol += int(np.sum(d < -1e-12))
    return {"arb_free": bool(worst >= -1e-10), "worst_violation": worst,
            "n_pairs": len(s) - 1,
            "pct_violating": float(100.0 * viol / (n * (len(s) - 1)))}


# ---------------------------------------------------------------------------
def fit_svi_slice(k: np.ndarray, total_var: np.ndarray, T: float,
                  weights: Optional[np.ndarray] = None,
                  enforce_butterfly: bool = True) -> SVIParams:
    """Fit one maturity's smile by weighted least squares.

    WHAT WE MINIMISE AND WHY IT IS WEIGHTED
    ---------------------------------------
    We fit in total-variance space but weight the residuals. The default
    weighting is by VEGA: an at-the-money option's price is very sensitive to
    volatility, so its implied vol is well determined; a far out-of-the-money
    option has almost no vega, so its implied vol is a wildly noisy number
    inferred from a price that is mostly bid-ask spread. Fitting them equally
    lets the least-reliable data points drag the curve around.

    PARAMETER BOUNDS, AND THE ONE THAT IS NOT OBVIOUS
    -------------------------------------------------
        b >= 0, sigma > 0, |rho| < 1                 shape requirements
        a + b*sigma*sqrt(1 - rho^2) >= 0             w(k) >= 0 EVERYWHERE

    The last one is the minimum of w over k. Variance cannot be negative at any
    strike, including strikes you did not observe, so it must be imposed on the
    parameters rather than checked on the data.

    THE HARD PART IS THE INITIAL GUESS
    ----------------------------------
    The SVI objective is non-convex and has local minima that look fine and fit
    badly. We seed from the data's own geometry (level, slope, curvature) and
    then try a grid of (m, sigma) starts, keeping the best. This is much more
    reliable than one clever start.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)
    good = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[good], w[good]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[good]
        weights = weights / (np.sum(weights) + 1e-300) * weights.size
    else:
        weights = np.ones_like(k)

    if k.size < 5:
        return SVIParams(np.nan, np.nan, np.nan, np.nan, np.nan, T,
                         n_points=int(k.size), converged=False)

    SIGMA_FLOOR = 5e-3      # see note below

    def unpack(p):
        a, b_raw, rho_raw, m, sig_raw = p
        b = np.exp(b_raw)                                   # b > 0
        rho = np.tanh(rho_raw)                              # |rho| < 1
        # A hard FLOOR on sigma, not just positivity. As sigma -> 0 the SVI
        # smile degenerates into a sharp V with a kink at k = m, which can fit
        # a handful of noisy points beautifully and is complete nonsense as a
        # volatility smile (it implies an atom of probability at one strike).
        # It is an attractive local optimum, and without this floor the 6-month
        # slice in the test chain collapsed into exactly that: sigma = 0.0000
        # and an RMSE of 7.6 vol points while every other slice fitted to 0.1.
        sigma = SIGMA_FLOOR + np.exp(sig_raw)
        return a, b, rho, m, sigma

    def residual(p):
        a, b, rho, m, sigma = unpack(p)
        d = k - m
        model = a + b * (rho * d + np.sqrt(d * d + sigma * sigma))
        res = np.sqrt(weights) * (model - w)
        # Penalty enforcing w >= 0 at the smile's minimum.
        floor = a + b * sigma * np.sqrt(max(1e-12, 1 - rho ** 2))
        pen = 1e3 * min(floor, 0.0)
        extra = [pen]
        if enforce_butterfly:
            # Soft penalty on Durrleman's g over a grid. Soft rather than hard
            # because a hard constraint often makes the fit fail entirely on
            # noisy real data; we report the violation instead of pretending.
            kk = np.linspace(k.min() - 0.2, k.max() + 0.2, 60)
            dd = kk - m
            root = np.sqrt(dd * dd + sigma * sigma)
            ww = np.maximum(a + b * (rho * dd + root), 1e-10)
            w1 = b * (rho + dd / root)
            w2 = b * sigma ** 2 / root ** 3
            g = (1 - kk * w1 / (2 * ww)) ** 2 - (w1 ** 2 / 4) * (1 / ww + 0.25) + w2 / 2
            extra.append(30.0 * float(np.sum(np.minimum(g, 0.0) ** 2)))
        return np.concatenate([res, np.asarray(extra)])

    # ---- SEEDING: exploit the fact that SVI is LINEAR in three parameters ---
    #
    # Rewrite the model with y1 = k - m and y2 = sqrt((k-m)^2 + sigma^2):
    #
    #     w = a + (b*rho) * y1 + b * y2
    #
    # For any FIXED (m, sigma) that is an ordinary linear regression in
    # (a, b*rho, b), solvable exactly in closed form. So the genuinely
    # non-convex part of SVI calibration is only TWO-dimensional.
    #
    # This is the idea behind Zeliade's "quasi-explicit" SVI calibration. We
    # use it to build seeds: sweep a grid of (m, sigma), solve the inner linear
    # problem at each node, and start the full non-linear fit from the best few
    # nodes. Far more reliable than guessing five numbers at once, and it is
    # what fixed the degenerate slice.
    spread = float(max(k.max() - k.min(), 0.05))
    sw = np.sqrt(weights)

    def inner_linear(m0: float, s0: float):
        y1 = k - m0
        y2 = np.sqrt(y1 * y1 + s0 * s0)
        X = np.column_stack([np.ones_like(k), y1, y2]) * sw[:, None]
        try:
            coef, *_ = np.linalg.lstsq(X, w * sw, rcond=None)
        except np.linalg.LinAlgError:
            return np.inf, None
        a0, c1, c2 = coef
        # Reject nodes whose exact linear solution violates SVI's shape
        # constraints (b >= 0, |rho| < 1); the non-linear stage could not
        # reach them anyway.
        if c2 <= 1e-10 or abs(c1) >= c2:
            return np.inf, None
        resid = X @ coef - w * sw
        return float(resid @ resid), (a0, c2, c1 / c2, m0, s0)

    seeds = []
    for m0 in np.linspace(k.min(), k.max(), 9):
        for s0 in np.geomspace(0.02 * spread + 1e-3, 1.5 * spread, 9):
            cost, params = inner_linear(float(m0), float(s0))
            if params is not None:
                seeds.append((cost, params))
    seeds.sort(key=lambda x: x[0])

    w_min = float(np.min(w))
    fallback = [(np.inf, (max(w_min * 0.8, 1e-6), b0, -0.4, 0.0, s0 * spread + 1e-3))
                for b0 in (0.05, 0.3) for s0 in (0.1, 0.5)]

    best, best_cost = None, np.inf
    for _, (a0, b0, rho0, m0, s0) in (seeds[:6] + fallback):
        p0 = np.array([a0, np.log(max(b0, 1e-6)),
                       np.arctanh(np.clip(rho0, -0.95, 0.95)),
                       m0, np.log(max(s0 - SIGMA_FLOOR, 1e-4))])
        for method in ("lm", "trf"):
            try:
                r = least_squares(residual, p0, method=method, max_nfev=4000)
            except Exception:
                continue
            if r.cost < best_cost:
                best, best_cost = r, r.cost
    if best is None:
        return SVIParams(np.nan, np.nan, np.nan, np.nan, np.nan, T,
                         n_points=int(k.size), converged=False)

    a, b, rho, m, sigma = unpack(best.x)
    fit = SVIParams(a, b, rho, m, sigma, T, n_points=int(k.size),
                    converged=bool(best.success))
    # RMSE reported in VOL points, not variance -- that is what a trader reads.
    model_vol = fit.implied_vol(k)
    data_vol = np.sqrt(np.maximum(w, 1e-12) / T)
    fit.rmse_vol = float(np.sqrt(np.mean((model_vol - data_vol) ** 2)))
    return fit


# ---------------------------------------------------------------------------
@dataclass
class VolSurface:
    slices: List[SVIParams] = field(default_factory=list)
    spot: float = float("nan")
    # `r`, `q` here are the MEAN rate/yield across the whole chain -- a
    # convenience used only by implied_vol()/price() below when interpolating
    # at a T outside the calibrated grid. Calibration itself does NOT use
    # these two scalars: it uses `rate_curve`, the per-expiry (r_T, q_T) that
    # produced each slice's forward, so a real term structure (short rates
    # different from long rates) is respected where it actually matters. This
    # is a known, deliberate simplification for the convenience path only --
    # see docs/CONCEPTS.md for why a full term-structure interpolation was
    # not built out.
    r: float = 0.0
    q: float = 0.0
    rate_curve: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    butterfly: Dict[float, Dict] = field(default_factory=dict)
    calendar: Dict = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)

    def maturities(self) -> List[float]:
        return [s.T for s in sorted(self.slices, key=lambda x: x.T)]

    def implied_vol(self, K, T) -> float:
        """Interpolate the surface at an arbitrary (strike, maturity).

        Interpolation is LINEAR IN TOTAL VARIANCE between the two bracketing
        expiries, not linear in volatility. This is the standard convention and
        it is not a detail: total variance is the additive quantity, so linear
        interpolation in w preserves the calendar no-arbitrage ordering, while
        linear interpolation in sigma can break it.
        """
        s = sorted(self.slices, key=lambda x: x.T)
        F = float(forward(self.spot, T, self.r, self.q))
        k = float(np.log(K / F))
        Ts = np.array([x.T for x in s])
        if T <= Ts[0]:
            return float(s[0].implied_vol(k))
        if T >= Ts[-1]:
            return float(s[-1].implied_vol(k))
        j = int(np.searchsorted(Ts, T))
        lo, hi = s[j - 1], s[j]
        wl = float(lo.total_variance(k))
        wh = float(hi.total_variance(k))
        lam = (T - lo.T) / (hi.T - lo.T)
        w = (1 - lam) * wl + lam * wh
        return float(np.sqrt(max(w, 1e-12) / T))

    def price(self, K, T, option="call") -> float:
        return float(bs_price(self.spot, K, T, self.r,
                              self.implied_vol(K, T), self.q, option))

    def summary(self) -> str:
        lines = ["", "  SVI SURFACE", "  " + "-" * 108,
                 f"  {'T (yrs)':>9}{'n':>5}{'out':>5}{'a':>10}{'b':>9}{'rho':>9}{'m':>9}"
                 f"{'sigma':>9}{'RMSE(vol%)':>12}{'ATM vol':>10}{'min g':>10}{'butterfly':>11}{'rho trust':>11}"]
        for s in sorted(self.slices, key=lambda x: x.T):
            bf = self.butterfly.get(s.T, {})
            lines.append(
                f"  {s.T:>9.4f}{s.n_points:>5}{s.n_outliers_dropped:>5}{s.a:>10.5f}{s.b:>9.4f}{s.rho:>9.4f}"
                f"{s.m:>9.4f}{s.sigma:>9.4f}{s.rmse_vol*100:>12.3f}"
                f"{float(s.implied_vol(0.0))*100:>9.2f}%"
                f"{bf.get('min_g', float('nan')):>10.4f}"
                f"{'OK' if bf.get('arb_free') else 'VIOLATION':>11}"
                f"{'':>1}{'ok' if s.reliable() else 'EXTRAPOL.':>10}")
        c = self.calendar
        kr = c.get("k_range_checked")
        lines += ["  " + "-" * 108,
                  f"  calendar arbitrage: "
                  f"{'none detected' if c.get('arb_free') else 'VIOLATION'}"
                  f"   worst dw = {c.get('worst_violation', float('nan')):.3e}"
                  f"   over {c.get('n_pairs', 0)} adjacent maturity pairs"
                  f"   checked over k in [{kr[0]:+.2f}, {kr[1]:+.2f}]" if kr else ""]
        return "\n".join(lines)


def build_surface(chain, spot: float, r: float = 0.0, q: float = 0.0,
                  min_points: int = 6, weight: str = "vega",
                  robust: bool = True, outlier_mad: float = 4.0) -> VolSurface:
    """Turn a cleaned option chain into a fitted, checked SVI surface.

    `chain` must be a dict of numpy arrays with keys:
        strike, expiry_years, option_type ('c'/'p'), mid_price
    and optionally vega, bid, ask.

    `r` and `q` may each be a single number shared by every contract, or one
    value PER CONTRACT (e.g. the output of data.apply_forward_rate, which
    assigns each expiry its own market-implied rate and yield). Both shapes
    are broadcast to match `K` here, so a live-chain caller passing a real
    term structure and a synthetic-chain caller passing one constant use
    exactly the same code path below.
    """
    K = np.asarray(chain["strike"], dtype=float)
    T = np.asarray(chain["expiry_years"], dtype=float)
    typ = np.asarray(chain["option_type"])
    px = np.asarray(chain["mid_price"], dtype=float)

    r_arr = np.broadcast_to(np.asarray(r, dtype=float), K.shape)
    q_arr = np.broadcast_to(np.asarray(q, dtype=float), K.shape)

    iv, conv, meth, iters = implied_vol_vector(px, spot, K, T, r_arr, q_arr, typ)

    surf = VolSurface(spot=spot, r=float(np.mean(r_arr)), q=float(np.mean(q_arr)))
    surf.diagnostics = {
        "n_quotes": int(K.size),
        "n_iv_converged": int(np.sum(conv)),
        "iv_convergence_pct": float(100.0 * np.mean(conv)),
        "n_newton": int(np.sum(meth == 0)),
        "n_brent_fallback": int(np.sum(meth == 1)),
        "n_failed": int(np.sum(meth == -1)),
        "mean_newton_iterations": float(np.mean(iters[meth == 0])) if np.any(meth == 0) else float("nan"),
    }

    for t in np.unique(T):
        sel = (T == t) & conv & np.isfinite(iv) & (iv > 1e-4)
        if int(np.sum(sel)) < min_points:
            continue
        r_t = float(np.mean(r_arr[T == t]))
        q_t = float(np.mean(q_arr[T == t]))
        surf.rate_curve[float(t)] = (r_t, q_t)
        F = float(forward(spot, t, r_t, q_t))
        k = np.log(K[sel] / F)
        w = iv[sel] ** 2 * t

        if weight == "vega" and "vega" in chain:
            wt = np.asarray(chain["vega"], dtype=float)[sel]
        elif weight == "vega":
            from .black_scholes import bs_greeks
            wt = np.asarray(bs_greeks(spot, K[sel], t, r_arr[sel], iv[sel], q_arr[sel], "call")["vega"])
        else:
            wt = np.ones_like(k)
        wt = np.maximum(wt, 1e-8)

        s = fit_svi_slice(k, w, float(t), wt)

        # ---- ROBUST SECOND PASS ----------------------------------------
        # Least squares assumes Gaussian errors and has a breakdown point of
        # zero: one arbitrarily bad point can move the fit arbitrarily far.
        # Real option chains always contain a few such points that survive
        # every mechanical filter.
        #
        # So: fit once, measure each point's residual, discard anything beyond
        # `outlier_mad` robust standard deviations, and refit on what is left.
        # We scale by the MEDIAN ABSOLUTE DEVIATION rather than the standard
        # deviation, because the standard deviation is itself inflated by the
        # outliers we are trying to find. (1.4826 * MAD estimates sigma for
        # Gaussian data.) This is trimmed least squares, and it is what makes
        # the difference between a surface and a plot on real data.
        n_dropped = 0
        if robust and np.isfinite(s.a) and k.size >= min_points + 2:
            resid = s.implied_vol(k) - np.sqrt(np.maximum(w, 1e-12) / t)
            mad = float(np.median(np.abs(resid - np.median(resid))))
            scale_r = max(1.4826 * mad, 1e-4)
            keep = np.abs(resid - np.median(resid)) <= outlier_mad * scale_r
            n_dropped = int(np.sum(~keep))
            if 0 < n_dropped <= max(2, k.size // 4) and int(np.sum(keep)) >= min_points:
                s = fit_svi_slice(k[keep], w[keep], float(t), wt[keep])

        if np.isfinite(s.a):
            s.n_outliers_dropped = n_dropped
            s.k_lo, s.k_hi = float(k.min()), float(k.max())
            surf.slices.append(s)
            surf.butterfly[float(t)] = check_butterfly(s, (s.k_lo - 0.3, s.k_hi + 0.3))

    # The calendar check compares slices at SHARED k values, so it needs a
    # range every slice can actually speak to. A fixed wide default (e.g.
    # +/-1.0) works on the synthetic chain, where every maturity is generated
    # over the same moneyness grid, but real chains are not so cooperative: a
    # short-dated slice's liquid strikes routinely cover a much narrower (and
    # often one-sided) k window than a long-dated one, because far-OTM short-
    # dated options round to a single tick and get filtered out entirely (see
    # clean_chain). Comparing a well-observed slice against another slice's
    # pure extrapolation produces "violations" that are really just "we asked
    # a curve about strikes it never saw" -- not a pricing inconsistency. So:
    # use the INTERSECTION of what every slice actually observed, and only
    # fall back to a fixed range if that intersection is empty (e.g. a single
    # slice, or two slices with no overlapping coverage at all).
    if len(surf.slices) >= 2:
        lo = max(s.k_lo - 0.3 for s in surf.slices if np.isfinite(s.k_lo))
        hi = min(s.k_hi + 0.3 for s in surf.slices if np.isfinite(s.k_hi))
        cal_range = (lo, hi) if lo < hi else (-1.0, 1.0)
    else:
        cal_range = (-1.0, 1.0)
    surf.calendar = check_calendar(surf.slices, k_range=cal_range)
    surf.calendar["k_range_checked"] = cal_range
    return surf
