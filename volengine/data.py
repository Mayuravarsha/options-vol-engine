"""Data: a synthetic option chain with KNOWN ground truth, plus real loaders.

WHY A SYNTHETIC CHAIN IS THE MORE USEFUL ONE
--------------------------------------------
With real market data you can fit a surface, but you can never tell whether the
fit is right, because the true volatility surface does not exist as an
observable object. With a synthetic chain generated from an SVI surface YOU
chose, you know every true parameter. So you can ask the question that actually
matters:

    Given noisy, discretely-quoted, bid-ask-spread-contaminated prices, does my
    pipeline recover the surface that generated them?

If it cannot do that, nothing it says about real data is trustworthy. This is
the same logic as simulate_garch in garch.py: validate against known truth
first, then go to the market.

The generator deliberately includes the things that make real chains hard:

  * bid-ask spreads that widen for out-of-the-money and short-dated contracts
  * prices rounded to a real tick grid ($0.01 below $3, $0.05 above)
  * a hard price floor -- deep out-of-the-money options quote at $0.01 or
    $0.05, and that rounding alone destroys their implied volatility
  * thin volume and open interest in the wings
  * a few genuinely stale/crossed quotes, because real feeds contain them

REAL DATA
---------
`load_yfinance_chain` pulls a live chain from Yahoo Finance if the optional
`yfinance` package is installed. It is imported lazily so the rest of the
project never depends on it.
"""

from __future__ import annotations

import csv as _csv
import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .black_scholes import bs_greeks, bs_price, forward


# ---------------------------------------------------------------------------
# A realistic equity-index-like SVI surface used as ground truth.
# rho is strongly negative: that IS the equity skew.
DEFAULT_TRUE_SVI: Dict[float, Dict[str, float]] = {
    #  T (years):  a        b       rho      m        sigma
    0.0192: dict(a=0.00035, b=0.0290, rho=-0.72, m=0.0060, sigma=0.0300),  # 1 week
    0.0833: dict(a=0.00160, b=0.0620, rho=-0.68, m=0.0110, sigma=0.0620),  # 1 month
    0.2500: dict(a=0.00560, b=0.1180, rho=-0.62, m=0.0180, sigma=0.1050),  # 3 months
    0.5000: dict(a=0.01250, b=0.1750, rho=-0.58, m=0.0250, sigma=0.1500),  # 6 months
    1.0000: dict(a=0.02700, b=0.2500, rho=-0.52, m=0.0350, sigma=0.2100),  # 1 year
    2.0000: dict(a=0.06000, b=0.3400, rho=-0.45, m=0.0500, sigma=0.3000),  # 2 years
}


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def true_iv(K, T, S, r, q, params: Optional[Dict] = None) -> np.ndarray:
    """Ground-truth implied volatility from the reference surface."""
    p = (params or DEFAULT_TRUE_SVI)[T]
    F = forward(S, T, r, q)
    k = np.log(np.asarray(K, dtype=float) / F)
    w = svi_total_variance(k, **p)
    return np.sqrt(np.maximum(w, 1e-10) / T)


def _round_to_tick(px: np.ndarray) -> np.ndarray:
    """US option ticks: $0.01 below $3.00, $0.05 at or above."""
    px = np.asarray(px, dtype=float)
    return np.where(px < 3.0, np.round(px / 0.01) * 0.01, np.round(px / 0.05) * 0.05)


def generate_synthetic_chain(S: float = 100.0, r: float = 0.042, q: float = 0.013,
                             maturities: Optional[Sequence[float]] = None,
                             n_strikes: int = 25,
                             moneyness_range: Tuple[float, float] = (0.70, 1.35),
                             seed: int = 2024,
                             params: Optional[Dict] = None,
                             noise_vol_pts: float = 0.0035) -> Dict[str, np.ndarray]:
    """Build a realistic, messy option chain from a known SVI surface.

    Returns a dict of parallel numpy arrays, plus `true_iv` so downstream code
    can score itself. `noise_vol_pts` is idiosyncratic noise in VOLATILITY
    points added before pricing (0.0035 = 0.35 vol points), on top of the
    spread and tick effects.
    """
    rng = np.random.default_rng(seed)
    params = params or DEFAULT_TRUE_SVI
    Ts = list(maturities) if maturities is not None else sorted(params)

    rows: List[Dict] = []
    for T in Ts:
        F = float(forward(S, T, r, q))
        strikes = np.round(np.linspace(moneyness_range[0] * S,
                                       moneyness_range[1] * S, n_strikes), 1)
        iv_true = true_iv(strikes, T, S, r, q, params)
        iv_noisy = np.maximum(iv_true + rng.normal(0, noise_vol_pts, strikes.size), 0.01)

        for K, ivt, ivn in zip(strikes, iv_true, iv_noisy):
            # Convention: quote the OUT-OF-THE-MONEY side. That is what is
            # liquid in reality, and it avoids the deep-ITM contracts whose
            # value is nearly all intrinsic and whose IV is therefore an
            # extremely noisy number backed out of a tiny time value.
            otype = "c" if K >= F else "p"
            fair = float(bs_price(S, K, T, r, ivn, q, otype))

            vega = float(bs_greeks(S, K, T, r, ivn, q, otype)["vega"])
            # Spread widens with moneyness distance and shortens with maturity.
            moneyness_dist = abs(np.log(K / F))
            rel_spread = 0.02 + 0.55 * moneyness_dist + 0.010 / np.sqrt(max(T, 1e-3))
            half = max(0.005, 0.5 * rel_spread * max(fair, 0.02))

            bid = _round_to_tick(max(fair - half, 0.0))
            ask = _round_to_tick(fair + half)
            tick = 0.01 if fair < 3.0 else 0.05
            if ask <= bid:
                ask = bid + tick
            # Deep wings genuinely trade at the minimum tick.
            if fair < 0.02:
                bid, ask = 0.0, tick

            liquidity = np.exp(-6.0 * moneyness_dist ** 2)
            volume = int(rng.poisson(max(0.2, 900 * liquidity * min(1.0, 0.4 / max(T, 0.05)))))
            oi = int(rng.poisson(max(1.0, 6000 * liquidity)))

            rows.append(dict(strike=float(K), expiry_years=float(T),
                             option_type=otype, bid=float(bid), ask=float(ask),
                             mid_price=float(0.5 * (bid + ask)),
                             true_iv=float(ivt), fair_price=fair,
                             vega=vega, volume=volume, open_interest=oi,
                             forward=F, stale=False))

    # Inject a handful of stale / crossed quotes -- real feeds have them and a
    # pipeline that cannot survive them is not a pipeline.
    n_bad = max(2, len(rows) // 60)
    for i in rng.choice(len(rows), n_bad, replace=False):
        if rng.random() < 0.5:
            rows[i]["bid"], rows[i]["ask"] = rows[i]["ask"], rows[i]["bid"]  # crossed
        else:
            rows[i]["mid_price"] *= float(rng.uniform(2.0, 4.0))            # stale
        rows[i]["stale"] = True

    out = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    out["spot"] = S
    out["r"] = r
    out["q"] = q
    return out


# ---------------------------------------------------------------------------
def clean_chain(chain: Dict[str, np.ndarray],
                max_rel_spread: float = 0.60,
                min_price: float = 0.05,
                min_volume: int = 0,
                min_open_interest: int = 1,
                moneyness_band: Tuple[float, float] = (-1.2, 1.0),
                min_days: float = 3.0,
                otm_only: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Filter an option chain. THIS STEP DECIDES THE QUALITY OF YOUR SURFACE.

    Every filter here removes a specific, real pathology:

    CROSSED OR ZERO QUOTES      bid > ask, or bid = ask = 0. Broken data.
    MID OUTSIDE THE SPREAD      a stale mid next to a live bid/ask. Looks like
                                a normal price; is not one.
    PRICE BELOW A FLOOR         a $0.02 option is one tick wide. Its implied
                                vol changes by several vol POINTS if the quote
                                moves one cent, so it carries no information
                                and enormous noise.
    WIDE RELATIVE SPREAD        if the spread is 60% of the mid, the "price" is
                                a fiction. The true value is somewhere in a
                                huge interval and the mid is an arbitrary
                                choice inside it.
    NO OPEN INTEREST            a contract nobody holds has a quote nobody has
                                any incentive to keep accurate.
    EXTREME MONEYNESS           beyond roughly +/- 1 in log-moneyness the
                                option is worth pennies and its IV is noise.
    NEAR EXPIRY                 under a few days, time value collapses,
                                pin risk dominates, and IV is unstable.
    OTM ONLY                    for in-the-money options, most of the price is
                                intrinsic value, so the IV is inferred from a
                                small residual and is correspondingly noisy.
                                It also sidesteps early-exercise effects for
                                American options. Standard practice, and it is
                                justified by put-call parity: the OTM put and
                                the ITM call carry the same information, so you
                                lose nothing by keeping the liquid one.

    Returns (filtered_chain, counts_removed_by_reason). Always look at those
    counts. If a filter is removing most of your data, either the filter or
    the data is wrong, and you need to know which.
    """
    n = chain["strike"].size
    keep = np.ones(n, dtype=bool)
    reasons: Dict[str, int] = {}

    def drop(mask: np.ndarray, name: str) -> None:
        nonlocal keep
        newly = keep & mask
        reasons[name] = int(np.sum(newly))
        keep = keep & ~mask

    bid = chain.get("bid", chain["mid_price"])
    ask = chain.get("ask", chain["mid_price"])
    mid = chain["mid_price"]

    drop(~np.isfinite(mid) | (mid <= 0), "non-positive or NaN mid")
    drop(ask < bid, "crossed quote (bid > ask)")
    # The mid must actually lie between the bid and the ask. This sounds too
    # obvious to be worth checking -- but a stale mid paired with a live
    # bid/ask is one of the most common defects in real feeds, and it is
    # invisible to every other filter here because the price on its own still
    # looks perfectly plausible. In testing, ONE such quote near the money
    # (implying 71% vol against a true 31%) was enough to throw an entire
    # maturity slice off by 7 volatility points, because vega weighting gives
    # at-the-money quotes the most influence over the fit.
    tol = 0.02 + 0.02 * np.abs(mid)
    drop((mid < bid - tol) | (mid > ask + tol), "mid outside [bid, ask]")
    drop(bid <= 0, "zero bid")
    drop(mid < min_price, f"price below ${min_price:.2f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mid > 0, (ask - bid) / mid, np.inf)
    drop(rel > max_rel_spread, f"relative spread > {max_rel_spread:.0%}")

    if "volume" in chain and min_volume > 0:
        drop(chain["volume"] < min_volume, f"volume < {min_volume}")
    if "open_interest" in chain and min_open_interest > 0:
        drop(chain["open_interest"] < min_open_interest,
             f"open interest < {min_open_interest}")

    drop(chain["expiry_years"] * 365.0 < min_days, f"under {min_days:.0f} days to expiry")

    F = chain.get("forward")
    if F is None:
        F = forward(chain["spot"], chain["expiry_years"], chain["r"], chain["q"])
    k = np.log(chain["strike"] / F)
    drop((k < moneyness_band[0]) | (k > moneyness_band[1]),
         f"log-moneyness outside [{moneyness_band[0]}, {moneyness_band[1]}]")

    if otm_only:
        is_call = np.isin(chain["option_type"], ["c", "call", "C"])
        itm = (is_call & (chain["strike"] < F)) | (~is_call & (chain["strike"] > F))
        drop(itm, "in-the-money (OTM convention)")

    out = {key: (val[keep] if isinstance(val, np.ndarray) and val.size == n else val)
           for key, val in chain.items()}
    reasons["KEPT"] = int(np.sum(keep))
    reasons["TOTAL"] = n
    return out, reasons


# ---------------------------------------------------------------------------
def load_yfinance_chain(ticker: str = "SPY", r: float = 0.042, q: float = 0.013,
                        max_expiries: int = 8) -> Optional[Dict[str, np.ndarray]]:
    """Pull a live option chain from Yahoo Finance, if `yfinance` is installed.

    Returns None (with an explanation printed) if the package is missing or the
    network is unavailable, so nothing else in the project depends on it.

    CAVEATS FOR REAL DATA, WHICH THE SYNTHETIC PATH DOES NOT HAVE
    -------------------------------------------------------------
    * Quotes are delayed and are NOT synchronised with the spot price you fetch
      separately. A stale spot alone can shift the whole smile.
    * `r` and `q` are assumed, not observed. A wrong dividend yield tilts the
      surface, and for single names the borrow cost matters too. Professionals
      back BOTH out of put-call parity per expiry rather than assuming them.
    * US equity options are AMERICAN. Black-Scholes prices European options.
      The early-exercise premium is small for OTM options on low-dividend names
      -- which is another reason to use the OTM convention -- but it is not
      zero, and it is largest for in-the-money puts.
    * After hours, many quotes are stale or crossed. The cleaning filters above
      exist for exactly this.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance is not installed. Install with `pip install yfinance`, "
              "or use generate_synthetic_chain(), which needs no network.")
        return None

    import datetime as dt
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            print(f"  no price history returned for {ticker}")
            return None
        spot = float(hist["Close"].iloc[-1])
        today = dt.date.today()

        rows: List[Dict] = []
        for exp in tk.options[:max_expiries]:
            exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
            T = max((exp_date - today).days, 1) / 365.0
            oc = tk.option_chain(exp)
            for df, otype in ((oc.calls, "c"), (oc.puts, "p")):
                for _, row in df.iterrows():
                    b, a = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
                    rows.append(dict(
                        strike=float(row["strike"]), expiry_years=T,
                        option_type=otype, bid=b, ask=a,
                        mid_price=0.5 * (b + a),
                        volume=int(row.get("volume", 0) or 0),
                        open_interest=int(row.get("openInterest", 0) or 0),
                    ))
        if not rows:
            print(f"  no option rows returned for {ticker}")
            return None
        out = {k: np.array([row[k] for row in rows]) for k in rows[0]}
        out["spot"], out["r"], out["q"] = spot, r, q
        out["forward"] = forward(spot, out["expiry_years"], r, q)
        out["ticker"] = ticker
        return out
    except Exception as exc:
        print(f"  could not fetch {ticker} from Yahoo Finance: {exc}")
        return None


def load_cached_chain(path: str) -> Dict[str, np.ndarray]:
    """Load one of our frozen CSV snapshots (see scripts/fetch_real_chains.py).

    A live chain is not a fixed dataset -- it changes every few seconds and
    the contracts themselves expire, so calibrating against the network every
    time would make the project impossible to reproduce: run it today and run
    it again next month and two different datasets produced two different
    surfaces, with no record of which. `fetch_real_chains.py` pulls the chain
    exactly once and freezes it as a CSV with the snapshot date written into
    every row; this function reads that file back into the same chain-dict
    shape `generate_synthetic_chain` and `load_yfinance_chain` produce, so
    every downstream function (clean_chain, build_surface, ...) treats a
    real, cached chain identically to a live or synthetic one.

    `expiry_years` is recomputed here from the two dates stored in the file
    (snapshot_date, expiry_date), not from "today" -- so the chain means the
    same thing whether you load it five minutes or five years after it was
    fetched. That is the actual content of "reproducible": not that the data
    is fake, but that it is dated and nailed down.
    """
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")

    snap_date = dt.date.fromisoformat(rows[0]["snapshot_date"])
    ticker = rows[0]["ticker"]
    spot = float(rows[0]["spot"])

    strike, T, otype, bid, ask, vol, oi, iv_y = [], [], [], [], [], [], [], []
    for row in rows:
        exp_date = dt.date.fromisoformat(row["expiry_date"])
        days = max((exp_date - snap_date).days, 1)
        strike.append(float(row["strike"]))
        T.append(days / 365.0)
        otype.append(row["option_type"])
        bid.append(float(row["bid"]))
        ask.append(float(row["ask"]))
        vol.append(int(float(row["volume"])))
        oi.append(int(float(row["open_interest"])))
        iv_y.append(float(row.get("implied_vol_yahoo", "nan") or "nan"))

    bid_a, ask_a = np.array(bid), np.array(ask)
    return {
        "strike": np.array(strike), "expiry_years": np.array(T),
        "option_type": np.array(otype), "bid": bid_a, "ask": ask_a,
        "mid_price": 0.5 * (bid_a + ask_a), "volume": np.array(vol),
        "open_interest": np.array(oi), "implied_vol_yahoo": np.array(iv_y),
        "spot": spot, "ticker": ticker, "snapshot_date": rows[0]["snapshot_date"],
    }


def implied_forward_rate(chain: Dict[str, np.ndarray], spot: float,
                         assumed_r: float = 0.042, moneyness_band: float = 0.15,
                         min_pairs: int = 3, fallback_q: float = 0.0
                         ) -> Dict[float, Dict[str, float]]:
    """Per expiry, back the FORWARD -- and hence the dividend/borrow yield q
    -- out of put-call parity, instead of assuming a dividend yield.

    THE IDENTITY, AND WHY IT IS MODEL-FREE
    ---------------------------------------
    For a call and a put at the SAME strike K and expiry T:

        C - P = D * (F - K)          D = e^{-rT} the discount factor
                                      F the forward price of the stock

    This holds under no-arbitrage ALONE. It does not assume Black-Scholes, a
    volatility, or any distribution for the stock -- it is true because a long
    call + short put has exactly the payoff of a forward contract, whatever
    model you believe. Rearranged per strike:  F = (C - P)/D + K.

    WHY r IS FIXED RATHER THAN ALSO EXTRACTED -- A REAL FINDING, NOT A GUESS
    --------------------------------------------------------------------------
    The first version of this function fit D AND F simultaneously, as a linear
    regression of (C - P) against K (slope -D, intercept D*F). On the real SPY
    chain that produced nonsense: the per-expiry rate came back swinging from
    -23% to +21% across adjacent weekly expiries, while the forward it also
    produced stayed put at 762-764 (spot was 761.69) the whole time.

    The reason is a signal-to-noise problem, not a bug. At short T,
    D = e^{-rT} is extremely close to 1 -- e.g. e^{-0.04 * 5/365} = 0.99945 --
    so the regression's SLOPE has to detect a rate signal smaller than 0.1%,
    while the bid-ask spread on these quotes alone is noise of a similar or
    larger size. D (and therefore r on its own) is simply not identified by a
    week's worth of option prices: there isn't enough time value of money in
    five days for the market to reveal what r is. F is a different story --
    it is pinned by WHERE the (C - P) line crosses zero, which is a robust
    geometric feature of the data regardless of the line's exact slope, so it
    stays well-determined even while D does not.

    The fix mirrors what a trading desk actually does: FIX r at a known,
    externally-observed short-term rate (`assumed_r` -- e.g. the prevailing
    T-bill/SOFR rate; this project uses 0.042 throughout, e.g. in
    generate_synthetic_chain), and solve for only the ONE quantity the chain
    can actually tell you: F. With D = e^{-assumed_r * T} now fixed and known,
    each liquid strike gives an independent estimate F_i = (C_i - P_i)/D + K_i;
    F_T is their spread-weighted average, and q_T = assumed_r - ln(F_T/S)/T is
    the market-implied dividend/borrow yield -- now the one thing this
    function actually discovers, rather than two things it was guessing at
    once. `f_std` in the return value reports the dispersion across strikes as
    a fit-quality diagnostic: a wide spread means the "liquid" pairs disagree
    about F and the result should be trusted less.

    THE ALTERNATIVE OF A FIXED q TOO, AND WHY THAT ONE WAS REJECTED
    -------------------------------------------------------------------
    Assuming q as well (this project's ORIGINAL approach, see
    load_yfinance_chain's constant r/q) is simpler still, but q is exactly the
    number that varies most across names and is least observable externally:
    it mixes the trailing dividend yield with the stock's SHORT-SELL BORROW
    COST, which is not public data and can be large and time-varying for a
    heavily-shorted name. r, by contrast, IS public (a T-bill or SOFR print)
    and nearly the same for every US equity on the same day. So fixing the
    number that is genuinely observable and solving for the number that
    genuinely is not is the right split, not an arbitrary one.

    Returns {T: {"F", "r", "q", "D", "n_pairs", "f_std", "method"}}. An expiry
    with fewer than `min_pairs` usable (call, put) strikes within
    `moneyness_band` of spot falls back to (`assumed_r`, `fallback_q`) with
    method="assumed", so a thin real chain degrades gracefully instead of
    returning nonsense.
    """
    K = np.asarray(chain["strike"], dtype=float)
    T = np.asarray(chain["expiry_years"], dtype=float)
    typ = np.asarray(chain["option_type"])
    mid = np.asarray(chain["mid_price"], dtype=float)
    bid = np.asarray(chain.get("bid", mid), dtype=float)
    ask = np.asarray(chain.get("ask", mid), dtype=float)
    is_call = np.isin(typ, ["c", "call", "C"])

    def _assumed(t: float, n_pairs: int, why: str) -> Dict[str, float]:
        D = float(np.exp(-assumed_r * t))
        return dict(F=float(spot * np.exp((assumed_r - fallback_q) * t)),
                    r=assumed_r, q=fallback_q, D=D, n_pairs=n_pairs,
                    f_std=float("nan"), method=f"assumed ({why})")

    curve: Dict[float, Dict[str, float]] = {}
    for t in np.unique(T):
        band = ((T == t)
               & (np.abs(np.log(np.maximum(K, 1e-6) / spot)) <= moneyness_band))
        valid = band & (mid > 0) & (ask >= bid) & (bid > 0)
        calls = {float(K[i]): (mid[i], ask[i] - bid[i])
                for i in np.where(valid & is_call)[0]}
        puts = {float(K[i]): (mid[i], ask[i] - bid[i])
               for i in np.where(valid & ~is_call)[0]}
        common = sorted(set(calls) & set(puts))

        if len(common) < min_pairs:
            curve[float(t)] = _assumed(float(t), len(common), "too few liquid pairs")
            continue

        t = float(t)
        D = float(np.exp(-assumed_r * t))
        Ks = np.array(common)
        cp = np.array([calls[k][0] - puts[k][0] for k in common])
        spreads = np.array([calls[k][1] + puts[k][1] for k in common])
        F_i = cp / D + Ks                              # one estimate per strike
        w = 1.0 / np.maximum(spreads, 0.01)
        F = float(np.average(F_i, weights=w))
        f_std = float(np.sqrt(np.average((F_i - F) ** 2, weights=w)))

        r = assumed_r
        q = float(r - np.log(max(F, 1e-6) / spot) / t)
        curve[t] = dict(F=F, r=r, q=q, D=D, n_pairs=len(common), f_std=f_std,
                        method="parity (r assumed)")
    return curve


def apply_forward_rate(chain: Dict[str, np.ndarray],
                       curve: Dict[float, Dict[str, float]]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Broadcast a per-expiry (r, q) curve from `implied_forward_rate` onto
    per-CONTRACT arrays, matching the shape `implied_vol_vector` and
    `build_surface` expect (one rate pair per row, not one for the chain)."""
    T = np.asarray(chain["expiry_years"], dtype=float)
    r_arr = np.array([curve[float(t)]["r"] for t in T])
    q_arr = np.array([curve[float(t)]["q"] for t in T])
    return r_arr, q_arr


def load_returns_csv(path: str, column: str = "return") -> Optional[np.ndarray]:
    """Load a daily return series from a two-column CSV (date, return)."""
    if not os.path.exists(path):
        return None
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if column in (raw.dtype.names or ()):
        return np.asarray(raw[column], dtype=float)
    return np.asarray(raw[raw.dtype.names[-1]], dtype=float)


def load_yfinance_returns(ticker: str = "SPY", period: str = "10y") -> Optional[np.ndarray]:
    """Daily log returns from Yahoo Finance, if `yfinance` is available."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance is not installed; falling back to the bundled series.")
        return None
    try:
        px = yf.Ticker(ticker).history(period=period)["Close"].to_numpy(dtype=float)
        return np.diff(np.log(px))
    except Exception as exc:
        print(f"  could not fetch {ticker}: {exc}")
        return None
