"""Monte Carlo option pricing, with variance reduction that actually works.

=============================================================================
WHY MONTE CARLO WHEN A FORMULA EXISTS?
=============================================================================
For a European call it is pointless -- Black-Scholes gives the exact answer
instantly. We do it anyway for two reasons:

  1. VALIDATION. We know the right answer, so we can check that the simulation
     converges to it, that the error shrinks at the rate theory predicts, and
     that our variance-reduction tricks are correct rather than merely
     plausible. Building a Monte Carlo engine against a known answer first is
     how you earn the right to trust it on a problem with no known answer.

  2. Most real payoffs have no closed form. Asian options (payoff depends on
     the AVERAGE price), barriers, lookbacks, autocallables, anything
     path-dependent or multi-asset. Monte Carlo handles all of them with the
     same code. `mc_asian_call` below is the demonstration.

=============================================================================
THE METHOD
=============================================================================
Under the risk-neutral measure the stock follows geometric Brownian motion, and
the exact solution of that SDE is

    S_T = S_0 * exp( (r - q - sigma^2/2) T  +  sigma sqrt(T) Z ),   Z ~ N(0,1)

Note this is EXACT, not a discretisation. For a payoff that depends only on
S_T there is no time-stepping error at all -- one normal draw per path. (For
path-dependent payoffs you do need to step through time, which is why the Asian
pricer takes n_steps.)

Then:  price = e^{-rT} * average of payoff(S_T) over many paths.

=============================================================================
THE CENTRAL PROBLEM: sqrt(N) IS SLOW
=============================================================================
The standard error of a Monte Carlo estimate is  s / sqrt(N).

That sqrt is brutal. Ten times more accuracy costs ONE HUNDRED times more
paths. Getting from 1 cent of error to 0.1 cents means 100x the compute.

You cannot beat the sqrt(N) rate with plain Monte Carlo. What you CAN do is
shrink the numerator s. That is what variance reduction is, and it is the whole
craft. Two techniques are implemented here.

-----------------------------------------------------------------------------
1. ANTITHETIC VARIATES
-----------------------------------------------------------------------------
For every draw Z, also use -Z. Both are valid N(0,1) draws, so the estimator
stays unbiased. The pair is negatively correlated: when Z gives a high stock
price, -Z gives a low one, and their errors partially cancel.

    Var( (X + X') / 2 ) = ( Var(X) + Var(X') + 2 Cov(X, X') ) / 4

With Cov < 0 this is smaller than Var(X)/2, which is what you would get from
two independent draws. Free lunch -- but only a small one for a call, because
max(S_T - K, 0) is a CONVEX function of Z, and convexity partly undoes the
cancellation. Antithetics help most for payoffs that are close to linear (and
they can even HURT for strongly convex ones, which is worth knowing).

-----------------------------------------------------------------------------
2. CONTROL VARIATES -- the big one
-----------------------------------------------------------------------------
Suppose alongside the thing you want (the payoff) you can compute something
correlated whose exact mean you already know. Here: S_T itself, with the known
mean  E[S_T] = S_0 e^{(r-q)T}.

On each path you can see the error the simulation made on the control -- and
because the payoff is correlated with the control, you can infer part of the
error it made on the payoff, and subtract it:

    estimator = mean( payoff + c * (S_T - E[S_T]) )

Still unbiased, because the correction has mean zero by construction. The
optimal coefficient is

    c* = -Cov(payoff, S_T) / Var(S_T)

and it delivers

    Var_reduced = Var_original * (1 - rho^2)

where rho is the correlation between payoff and control. That is the formula
worth remembering: a correlation of 0.9 removes 81% of the variance; 0.99
removes 98%. For an in-the-money call, rho is very close to 1 because the
payoff is nearly S_T - K, so the reduction is enormous.

(Estimating c* from the same paths introduces a tiny O(1/N) bias. It is
negligible here and universally accepted in practice; the alternative is a
small pilot run to estimate c* separately.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price


@dataclass
class MCResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    variance: float
    method: str
    control_beta: float = float("nan")
    control_corr: float = float("nan")
    seconds: float = 0.0

    def __repr__(self) -> str:
        return (f"<MC {self.method}: {self.price:.6f} +/- {self.stderr:.6f} "
                f"(95% CI [{self.ci_low:.6f}, {self.ci_high:.6f}], "
                f"N={self.n_paths:,})>")


def _summarize(values: np.ndarray, method: str, **kw) -> MCResult:
    n = values.size
    mean = float(values.mean())
    var = float(values.var(ddof=1))
    se = float(np.sqrt(var / n))
    z = 1.959963984540054            # 97.5th percentile of N(0,1)
    return MCResult(price=mean, stderr=se, ci_low=mean - z * se,
                    ci_high=mean + z * se, n_paths=n, variance=var,
                    method=method, **kw)


# ---------------------------------------------------------------------------
def mc_european(S, K, T, r, sigma, q=0.0, option="call", n_paths: int = 200_000,
                antithetic: bool = True, control_variate: bool = True,
                seed: Optional[int] = 12345) -> MCResult:
    """Price a European option by simulating the terminal price only."""
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if antithetic:
        # Draw half the normals and mirror them, so the sample is exactly
        # symmetric about zero and the total path count is still n_paths.
        half = n_paths // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T)
    ST = S * np.exp(drift + diffusion * z)

    payoff = (np.maximum(ST - K, 0.0) if option.lower() in ("c", "call")
              else np.maximum(K - ST, 0.0))
    disc = np.exp(-r * T)
    discounted = disc * payoff

    beta = corr = float("nan")
    if control_variate:
        # Control: the discounted stock price. Its exact mean is known.
        control = disc * ST
        control_mean = S * np.exp(-q * T)           # = disc * S e^{(r-q)T}
        cov = np.cov(discounted, control, ddof=1)
        var_c = cov[1, 1]
        if var_c > 0:
            beta = -cov[0, 1] / var_c
            sd = np.sqrt(cov[0, 0] * var_c)
            corr = float(cov[0, 1] / sd) if sd > 0 else float("nan")
            discounted = discounted + beta * (control - control_mean)

    method = ("plain" if not (antithetic or control_variate) else
              "+".join(([" antithetic"] if antithetic else []) +
                       ([" control"] if control_variate else []))).strip()
    res = _summarize(discounted, method or "plain",
                     control_beta=beta, control_corr=corr)
    res.seconds = time.perf_counter() - t0
    return res


def geometric_asian_call(S, K, T, r, sigma, q=0.0, n: int = 52) -> float:
    """Exact price of a discretely-monitored GEOMETRIC-average Asian call.

    WHY A CLOSED FORM EXISTS HERE BUT NOT FOR THE ARITHMETIC AVERAGE
    ----------------------------------------------------------------
    Each S_t is lognormal, i.e. log S_t is normal. The geometric average is
    exp(mean of the logs) -- and a mean of NORMALS is normal, so the geometric
    average is itself exactly lognormal. Black-Scholes applies verbatim, with
    a modified drift and volatility.

    The ARITHMETIC average is a sum of lognormals, which has no closed-form
    distribution at all. Hence Monte Carlo, and hence the geometric version
    being such a good control variate: same paths, correlation above 0.99, and
    one of the two answers is known exactly.

    With monitoring at t_i = iT/n for i = 1..n, log G is normal with

        mean  = log S + (r - q - sigma^2/2) * T(n+1)/(2n)
        var   = sigma^2 * T * (n+1)(2n+1) / (6 n^2)
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sig_g = sigma * np.sqrt((n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n))
    mu_g = (r - q - 0.5 * sigma ** 2) * (n + 1.0) / (2.0 * n) + 0.5 * sig_g ** 2
    vt = sig_g * np.sqrt(T)
    d1 = (np.log(S / K) + (mu_g + 0.5 * sig_g ** 2) * T) / vt
    d2 = d1 - vt
    return float(np.exp(-r * T) * (S * np.exp(mu_g * T) * norm.cdf(d1)
                                   - K * norm.cdf(d2)))


def mc_asian_call(S, K, T, r, sigma, q=0.0, n_paths: int = 100_000,
                  n_steps: int = 52, antithetic: bool = True,
                  control_variate: bool = True,
                  seed: Optional[int] = 12345) -> MCResult:
    """Arithmetic-average Asian call -- the payoff with no closed form.

    Payoff = max( average(S_t) - K, 0 ), averaged over n_steps monitoring dates.

    WHY THIS IS THE INTERESTING CASE
    --------------------------------
    There is no Black-Scholes formula, because a sum of lognormals is not
    lognormal. (A sum of NORMALS is normal, which is why the GEOMETRIC-average
    Asian does have a closed form -- and that is exactly what we exploit below.)

    Asians are widely used in commodities and FX precisely because averaging
    makes the payoff much harder to manipulate near expiry and reduces the
    effect of a single spike.

    THE CONTROL VARIATE HERE IS THE ELEGANT PART
    --------------------------------------------
    We use the GEOMETRIC-average Asian, which is analytic, as the control for
    the ARITHMETIC one. They are correlated at well over 0.99 because the two
    averages differ only by a small Jensen gap, so the variance reduction is
    dramatic -- often 50-100x. This is a standard, well-known trick and a very
    good thing to be able to explain.
    """
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    log_paths = np.cumsum(drift + vol * z, axis=1)
    paths = S * np.exp(log_paths)

    arith = paths.mean(axis=1)
    disc = np.exp(-r * T)
    payoff = disc * np.maximum(arith - K, 0.0)

    beta = corr = float("nan")
    if control_variate:
        geo = np.exp(np.log(paths).mean(axis=1))
        geo_payoff = disc * np.maximum(geo - K, 0.0)
        geo_exact = geometric_asian_call(S, K, T, r, sigma, q, n_steps)

        cov = np.cov(payoff, geo_payoff, ddof=1)
        if cov[1, 1] > 0:
            beta = -cov[0, 1] / cov[1, 1]
            sd = np.sqrt(cov[0, 0] * cov[1, 1])
            corr = float(cov[0, 1] / sd) if sd > 0 else float("nan")
            payoff = payoff + beta * (geo_payoff - geo_exact)

    res = _summarize(payoff, "asian(antithetic+geometric control)"
                     if control_variate else "asian(plain)",
                     control_beta=beta, control_corr=corr)
    res.seconds = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
def mc_pathwise_delta(S, K, T, r, sigma, q=0.0, n_paths: int = 200_000,
                      seed: Optional[int] = 12345) -> Dict[str, float]:
    """Delta by the PATHWISE method, versus bumping.

    THE NAIVE WAY: run the whole simulation at S and again at S + h, and take
    the difference. Two problems. First, it costs two simulations. Second, and
    much worse, if you use different random numbers for the two runs the Monte
    Carlo noise swamps the small real difference -- your "delta" is mostly
    noise. (Using the SAME random numbers for both, called common random
    numbers, fixes most of that and is what any competent implementation does.)

    THE PATHWISE WAY: differentiate INSIDE the expectation. Since

        S_T = S * exp(...)        =>      dS_T/dS = S_T / S

    the chain rule on the payoff gives, for a call,

        d/dS max(S_T - K, 0) = 1{S_T > K} * S_T / S

    so delta = e^{-rT} E[ 1{S_T > K} * S_T / S ]. One simulation, no bump, no
    differencing noise, and it reuses the paths you already generated.

    The catch: this needs the payoff to be differentiable almost everywhere.
    It works for calls and puts. It does NOT work directly for a digital
    option, whose payoff is a step function -- there you need the likelihood-
    ratio method or a smoothed payoff instead. Knowing which method applies to
    which payoff is a real interview question.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    disc = np.exp(-r * T)

    pathwise = disc * (ST > K) * ST / S
    est = float(pathwise.mean())
    se = float(pathwise.std(ddof=1) / np.sqrt(n_paths))

    # Bump-and-revalue using COMMON random numbers (same z for both legs).
    h = S * 1e-4
    up = disc * np.maximum(S * (1 + 1e-4) * np.exp(
        (r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z) - K, 0.0)
    dn = disc * np.maximum(S * (1 - 1e-4) * np.exp(
        (r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z) - K, 0.0)
    bump = float((up - dn).mean() / (2 * h))

    from .black_scholes import bs_greeks
    exact = float(bs_greeks(S, K, T, r, sigma, q, "call")["delta"])
    return {"pathwise": est, "pathwise_stderr": se,
            "bump_common_random_numbers": bump, "analytic": exact,
            "pathwise_error": est - exact, "bump_error": bump - exact}


# ---------------------------------------------------------------------------
def convergence_study(S, K, T, r, sigma, q=0.0, option="call",
                      path_counts=(1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000),
                      seed: int = 7) -> Dict[str, list]:
    """Error and standard error versus N, for each variance-reduction setting.

    WHAT TO LOOK FOR IN THE OUTPUT
    ------------------------------
    * The standard error should fall like 1/sqrt(N): quadruple N, halve the
      error. On a log-log plot that is a straight line of slope -0.5. If your
      line has a different slope, something is wrong with the estimator.
    * The absolute error against Black-Scholes should sit inside about two
      standard errors roughly 95% of the time. If it does not, the estimator is
      biased -- which is a bug, not bad luck.
    * The variance-reduction FACTOR (ratio of variances) should be roughly
      constant in N. Variance reduction buys you a better constant, never a
      better rate.
    """
    exact = float(bs_price(S, K, T, r, sigma, q, option))
    rows = {"n": [], "exact": exact}
    for label, anti, ctrl in [("plain", False, False),
                              ("antithetic", True, False),
                              ("control", False, True),
                              ("antithetic+control", True, True)]:
        prices, errs, ses, variances = [], [], [], []
        for n in path_counts:
            res = mc_european(S, K, T, r, sigma, q, option, n_paths=n,
                              antithetic=anti, control_variate=ctrl,
                              seed=seed + n)
            prices.append(res.price)
            errs.append(abs(res.price - exact))
            ses.append(res.stderr)
            variances.append(res.variance)
        rows[label] = {"price": prices, "abs_error": errs,
                       "stderr": ses, "variance": variances}
    rows["n"] = list(path_counts)
    return rows
