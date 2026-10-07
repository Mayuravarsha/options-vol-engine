"""Black-Scholes-Merton pricing and Greeks, fully vectorised.

=============================================================================
WHAT AN OPTION IS
=============================================================================
A CALL gives you the RIGHT (not the obligation) to BUY the stock at a fixed
price K (the "strike") on a fixed date T (European exercise). A PUT gives you
the right to SELL at K.

At expiry the value is obvious:
    call payoff = max(S_T - K, 0)      put payoff = max(K - S_T, 0)
If the stock ends at $120 and you hold a $100 call, you buy at 100 and it is
worth 120, so the option is worth $20. If the stock ends at $90 you simply do
not exercise, and it is worth nothing. You can never lose more than you paid.

The hard question is what it is worth BEFORE expiry, when S_T is unknown.

=============================================================================
THE BLACK-SCHOLES IDEA, WITHOUT THE STOCHASTIC CALCULUS
=============================================================================
The insight that won a Nobel Prize is NOT "compute the expected payoff". It is
this: you can build a portfolio of the stock and cash that EXACTLY replicates
the option, if you rebalance continuously.

Suppose you sell a call and simultaneously buy Delta shares. Choose Delta so
that a small move in the stock changes both sides of your book by the same
amount. Then your position is instantaneously insensitive to the stock price.
It is riskless. And a riskless portfolio must earn exactly the risk-free rate,
or you could arbitrage the difference.

That single requirement pins down the option price uniquely. Two consequences
follow, and both surprise people:

  1. The stock's EXPECTED RETURN does not appear in the formula. Whether you
     think the stock will double or halve is irrelevant, because the hedge
     removes the directional exposure. Only its VOLATILITY matters.

  2. Pricing becomes an expectation under the "risk-neutral measure" -- a
     fictional world where every asset drifts at the risk-free rate r. This
     is a computational device, not a claim about reality.

=============================================================================
THE FORMULA
=============================================================================
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

where N is the standard normal CDF, r is the risk-free rate and q is the
continuous dividend yield.

HOW TO READ IT (this is what interviewers ask):

  * N(d2) is the risk-neutral PROBABILITY the option finishes in the money.
    So K e^{-rT} N(d2) is "what you expect to pay, discounted".

  * S e^{-qT} N(d1) is the discounted expected value of the stock you RECEIVE,
    conditional on exercising, times the probability of exercising. N(d1) is
    NOT a probability on its own -- it is a probability re-weighted by the
    stock's value in each state.

  * So the whole formula reads: (what you get) minus (what you pay), each
    weighted by how likely you are to get or pay it.

  * e^{-qT} on the stock term: holding a call does not pay you dividends, so
    the stock you are entitled to is worth less by exactly the dividends you
    forgo.

=============================================================================
THE ASSUMPTIONS -- AND WHICH ONE IS ACTUALLY WRONG
=============================================================================
  1. The stock follows geometric Brownian motion with CONSTANT volatility.
  2. No transaction costs; continuous, free rebalancing.
  3. Constant known risk-free rate; unlimited borrowing/lending.
  4. No arbitrage.
  5. European exercise (no early exercise).

Assumption 1 is the one that fails, and it fails in a specific, measurable way.
Real returns have fat tails and volatility clusters. The market knows this, so
it charges more for out-of-the-money options than constant-vol Black-Scholes
says it should. Back out the implied volatility strike by strike and you do not
get a flat line -- you get the VOLATILITY SMILE. Documenting and modelling that
failure is what the rest of this project is about.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]

# Tiny floors, used so that a zero-time or zero-vol option returns its
# intrinsic value instead of dividing by zero.
_EPS_T = 1e-12
_EPS_V = 1e-12


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = map(np.asarray, np.broadcast_arrays(S, K, T, r, sigma, q))
    S = S.astype(float); K = K.astype(float); T = T.astype(float)
    r = r.astype(float); sigma = sigma.astype(float); q = q.astype(float)

    vt = sigma * np.sqrt(np.maximum(T, _EPS_T))
    vt = np.maximum(vt, _EPS_V)
    d1 = (np.log(np.maximum(S, _EPS_V) / np.maximum(K, _EPS_V))
          + (r - q + 0.5 * sigma ** 2) * T) / vt
    return d1, d1 - vt, S, K, T, r, sigma, q, vt


def forward(S, T, r, q=0.0):
    """Forward price F = S e^{(r-q)T}.

    The price agreed today for delivery at T. It is the natural centre of the
    option-pricing problem: at-the-money is properly defined relative to the
    FORWARD, not the spot, and the volatility smile is naturally a function of
    log(K/F).
    """
    return np.asarray(S) * np.exp((np.asarray(r) - np.asarray(q)) * np.asarray(T))


def bs_price(S, K, T, r, sigma, q=0.0, option: str = "call"):
    """Black-Scholes-Merton price. Every argument broadcasts."""
    d1, d2, S, K, T, r, sigma, q, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option.lower() in ("c", "call"):
        px = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        intrinsic = np.maximum(S * disc_q - K * disc_r, 0.0)
    else:
        px = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        intrinsic = np.maximum(K * disc_r - S * disc_q, 0.0)

    # At T=0 or sigma=0 the option is worth exactly its intrinsic value; the
    # d1/d2 expressions are undefined there, so substitute explicitly.
    degenerate = (T <= _EPS_T) | (sigma <= _EPS_V)
    px = np.where(degenerate, intrinsic, px)
    return px if px.ndim else float(px)


def bs_greeks(S, K, T, r, sigma, q=0.0, option: str = "call") -> Dict[str, np.ndarray]:
    """All the Greeks at once.

    WHAT EACH GREEK IS, IN PLAIN LANGUAGE
    -------------------------------------
    DELTA  dV/dS. How much the option price moves per $1 move in the stock.
           Also the number of shares you must hold to hedge one option. A call
           delta runs from 0 (far out of the money) to 1 (deep in the money),
           and is about 0.5 at the money. Roughly -- and only roughly -- the
           risk-neutral probability of finishing in the money.

    GAMMA  d(Delta)/dS. How fast your hedge goes stale. This is THE number for
           an options market maker: high gamma means you must rebalance
           constantly, and every rebalance costs the spread. Gamma is largest
           at the money and close to expiry, which is exactly when short-dated
           at-the-money options become dangerous to be short.

    VEGA   dV/d(sigma). Sensitivity to volatility itself. Reported here per
           1.00 of vol (so divide by 100 for the usual "per vol point").
           Vega is largest for LONG-dated at-the-money options -- more time
           means more chance for volatility to matter.

    THETA  dV/dt. Time decay, reported per YEAR (divide by 365 for per day).
           Almost always negative if you are long an option: every day that
           passes is one less day for the stock to move your way. Theta and
           gamma are two sides of one coin -- you are paid theta for being
           short gamma, and you pay theta for being long it.

    RHO    dV/dr. Interest-rate sensitivity. Usually the least important for
           short-dated equity options, but it matters for LEAPS and for
           anything rate-sensitive.

    VANNA  d(Delta)/d(sigma) = d(Vega)/dS. Tells you how your hedge changes
           when volatility moves. Central to risk-managing a skew position.

    VOLGA  d(Vega)/d(sigma), also called vomma. Convexity in volatility. It is
           why a portfolio that is vega-neutral can still make or lose money
           on a big vol move.

    A NOTE ON SIGNS AND UNITS -- where people get burned
    ----------------------------------------------------
    Gamma and vega are identical for a call and a put on the same strike (put-
    call parity differs by a forward contract, which has no gamma and no vega).
    Delta, theta and rho are not. Vega here is per 1.00 of vol and theta is per
    year; every desk uses a different convention, so always state yours.
    """
    d1, d2, S, K, T, r, sigma, q, vt = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)
    is_call = option.lower() in ("c", "call")
    sqrtT = np.sqrt(np.maximum(T, _EPS_T))

    if is_call:
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf_d1 / (S * vt)
    vega = S * disc_q * pdf_d1 * sqrtT
    vanna = -disc_q * pdf_d1 * d2 / np.maximum(sigma, _EPS_V)
    volga = vega * d1 * d2 / np.maximum(sigma, _EPS_V)

    out = {"price": bs_price(S, K, T, r, sigma, q, option),
           "delta": delta, "gamma": gamma, "vega": vega,
           "theta": theta, "rho": rho, "vanna": vanna, "volga": volga}
    return {k: (v if np.ndim(v) else float(v)) for k, v in out.items()}


def put_call_parity_gap(call_px, put_px, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}).

    Put-call parity is a pure no-arbitrage identity: it holds for ANY model,
    because buying a call and selling a put replicates a forward exactly. So
    this residual is the single best sanity check on a pricer -- if it is not
    zero to machine precision, the code is wrong, not the model.

    On real market data the gap is NOT zero, and its size is informative: it
    absorbs the bid-ask spread, a wrong dividend assumption, a wrong rate, and
    (for American options) the early-exercise premium. Traders back the implied
    dividend or the implied borrow cost out of exactly this number.
    """
    S, K, T, r, q = np.broadcast_arrays(S, K, T, r, q)
    return (np.asarray(call_px) - np.asarray(put_px)
            - (S * np.exp(-q * T) - K * np.exp(-r * T)))


def finite_difference_greeks(S, K, T, r, sigma, q=0.0, option="call",
                             h_rel: float = 1e-4) -> Dict[str, float]:
    """Greeks by bumping inputs -- an independent check on the closed forms.

    Why bother when we have exact formulas? Because a typo in an analytic Greek
    is invisible: the number looks plausible and is simply wrong. Bumping the
    price function is slow but nearly impossible to get wrong, so it is the
    reference. tests/test_greeks.py asserts the two agree.

    Central differences ( [f(x+h) - f(x-h)] / 2h ) are used because their error
    is O(h^2) rather than O(h). h is scaled to each variable's magnitude, since
    a bump of 1e-4 means something very different for S=100 than for r=0.03.
    """
    f = lambda s, k, t, rr, v, qq: bs_price(s, k, t, rr, v, qq, option)

    hS = max(S * h_rel, 1e-6)
    hv = max(sigma * h_rel, 1e-8)
    hT = max(T * h_rel, 1e-8)
    hr = max(abs(r) * h_rel, 1e-8)

    return {
        "delta": (f(S + hS, K, T, r, sigma, q) - f(S - hS, K, T, r, sigma, q)) / (2 * hS),
        "gamma": (f(S + hS, K, T, r, sigma, q) - 2 * f(S, K, T, r, sigma, q)
                  + f(S - hS, K, T, r, sigma, q)) / (hS ** 2),
        "vega": (f(S, K, T, r, sigma + hv, q) - f(S, K, T, r, sigma - hv, q)) / (2 * hv),
        # theta = dV/dt, and t runs the opposite way to T, hence the minus.
        "theta": -(f(S, K, T + hT, r, sigma, q) - f(S, K, T - hT, r, sigma, q)) / (2 * hT),
        "rho": (f(S, K, T, r + hr, sigma, q) - f(S, K, T, r - hr, sigma, q)) / (2 * hr),
    }
