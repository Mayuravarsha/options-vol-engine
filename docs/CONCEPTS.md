# Everything in this project, explained from zero

No prior options knowledge assumed. This builds from "what is an option" to
"why the implied volatility surface is a picture of the market's risk-neutral
distribution, and why my GARCH forecast sits a full volatility point below it".

Read it once end to end, then read the code — every module has a long header
comment written for someone who has read this.

---

## Table of contents

1. [What an option is](#1)
2. [Why pricing one is hard](#2)
3. [Black–Scholes, without the stochastic calculus](#3)
4. [Reading the formula](#4)
5. [The Greeks](#5)
6. [How you know your Greeks are right](#6)
7. [Implied volatility: running the formula backwards](#7)
8. [Where Newton breaks](#8)
9. [The volatility smile](#9)
10. [SVI: parameterising the smile](#10)
11. [Arbitrage: the two conditions that must hold](#11)
12. [Cleaning an option chain](#12)
12b. [Real chains: extracting the forward and rate](#12b)
13. [Monte Carlo](#13)
14. [Variance reduction](#14)
15. [The payoff with no formula](#15)
16. [Greeks from Monte Carlo](#16)
17. [Volatility clustering and GARCH](#17)
18. [Estimating GARCH by maximum likelihood](#18)
19. [The leverage effect](#19)
20. [Forecasting, and how to evaluate it honestly](#20)
21. [Implied vs realised: the variance risk premium](#21)
22. [The bugs I hit, and what each one taught](#22)

---

<a name="1"></a>
## 1. What an option is

A **call option** gives you the *right*, but not the obligation, to **buy** a
stock at a fixed price `K` (the **strike**) on a fixed date `T`. A **put** gives
you the right to **sell** at `K`.

(These are *European* options — exercisable only at `T`. American options can be
exercised any time before, which matters and is discussed in §12.)

At expiry the value is obvious:

```
call payoff = max(S_T − K, 0)
put  payoff = max(K − S_T, 0)
```

If the stock finishes at $120 and you hold a $100 call, you exercise: buy at 100,
it's worth 120, so the option is worth $20. If the stock finishes at $90, you
just don't exercise, and it expires worthless. **You can never lose more than you
paid.** That asymmetry — unlimited upside, capped downside — is the entire point.

The hard question is what it is worth *before* expiry, when `S_T` is unknown.

---

<a name="2"></a>
## 2. Why pricing one is hard

The obvious approach — "compute the expected payoff and discount it" — needs the
probability distribution of `S_T`, which needs the stock's expected return, which
nobody agrees on. Two people with different views on the stock would get
different option prices, and there'd be no such thing as *the* price.

Black and Scholes' insight was to sidestep that question entirely.

---

<a name="3"></a>
## 3. Black–Scholes, without the stochastic calculus

**The replication argument.** You can build a portfolio of the stock and cash
that *exactly* replicates the option — if you rebalance continuously.

Sell a call. Simultaneously buy `Δ` shares. Choose `Δ` so that a small move in
the stock changes both sides of your book by the same amount. Now your position
is instantaneously insensitive to the stock price. It is **riskless**.

And a riskless portfolio must earn exactly the risk-free rate — otherwise you
could borrow at the risk-free rate, buy the portfolio, and pocket the difference
with no risk. That single no-arbitrage requirement pins the option price down
**uniquely**.

Two consequences surprise everyone:

**1. The stock's expected return does not appear in the formula.** Whether you
think the stock will double or halve is irrelevant, because the hedge removes the
directional exposure. Only its **volatility** matters. Two traders with opposite
views on the stock must agree on the option's price.

**2. Pricing becomes an expectation under the "risk-neutral measure"** — a
fictional world where every asset drifts at the risk-free rate `r`. This is a
computational device, not a claim about reality. It works because the hedge makes
your actual beliefs irrelevant, so you may as well use the most convenient set of
beliefs.

Keep the distinction between the **real-world measure P** (what will actually
happen) and the **risk-neutral measure Q** (what the market charges). §21 is
entirely about the gap between them.

---

<a name="4"></a>
## 4. Reading the formula

```
d1 = [ ln(S/K) + (r − q + σ²/2)·T ] / (σ√T)
d2 = d1 − σ√T

Call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

`N` is the standard normal CDF, `r` the risk-free rate, `q` the continuous
dividend yield.

**How to read it** — this is what interviewers ask:

- **`N(d2)` is the risk-neutral probability the option finishes in the money.**
  So `K·e^(−rT)·N(d2)` is "what you expect to pay, discounted to today".

- **`S·e^(−qT)·N(d1)` is the discounted expected value of the stock you receive**,
  conditional on exercising, times the probability of exercising. `N(d1)` is *not*
  a probability on its own — it is a probability re-weighted by the stock's value
  in each state.

- So the whole thing reads: **(what you get) − (what you pay)**, each weighted by
  how likely you are to get or pay it.

- **The `e^(−qT)` on the stock term**: holding a call does not pay you dividends,
  so the stock you're entitled to is worth less by exactly the dividends you
  forgo.

**The forward** `F = S·e^((r−q)T)` is the price agreed today for delivery at `T`.
It is the natural centre of the problem: "at the money" is properly defined
relative to the *forward*, not the spot, and the volatility smile is naturally a
function of `log(K/F)`.

### The assumptions, and which one is actually wrong

1. The stock follows geometric Brownian motion with **constant volatility**
2. No transaction costs; continuous, free rebalancing
3. Constant known risk-free rate; unlimited borrowing and lending
4. No arbitrage
5. European exercise

**Assumption 1 is the one that fails**, and it fails in a specific, measurable
way. Real returns have fat tails and volatility clusters. The market knows this
and charges more for out-of-the-money options than constant-vol Black–Scholes
says it should. Backing out the implied volatility strike by strike gives you not
a flat line but the **volatility smile** (§9). Documenting and modelling that
failure is what the rest of this project is about.

### Put-call parity — the one identity that holds in every model

```
C − P = S·e^(−qT) − K·e^(−rT)
```

Buying a call and selling a put replicates a forward contract exactly, so this
must hold *regardless of any model*. That makes it the best sanity check on a
pricer: if the residual isn't zero to machine precision, **the code is wrong, not
the model**. In this project it comes out at `0.0e+00`.

On *real* market data the gap is not zero, and its size is informative: it
absorbs the bid-ask spread, a wrong dividend assumption, a wrong rate, and (for
American options) the early-exercise premium. Traders back the implied dividend
or borrow cost out of exactly this number.

---

<a name="5"></a>
## 5. The Greeks

Partial derivatives of the option price. Each one is a risk you can hedge.

| Greek | Definition | In words |
|---|---|---|
| **Delta** | `∂V/∂S` | Price change per $1 move in the stock. Also the number of shares to hold as a hedge. |
| **Gamma** | `∂²V/∂S²` | How fast your hedge goes stale. |
| **Vega** | `∂V/∂σ` | Sensitivity to volatility itself. |
| **Theta** | `∂V/∂t` | Time decay. |
| **Rho** | `∂V/∂r` | Interest-rate sensitivity. |
| **Vanna** | `∂²V/∂S∂σ` | How delta moves when vol moves. |
| **Volga** | `∂²V/∂σ²` | Convexity in volatility. |

### What actually matters about each

**Delta** runs from 0 (far out of the money) to 1 (deep in the money), and is
about 0.5 at the money. It is roughly — and only roughly — the risk-neutral
probability of finishing in the money.

**Gamma** is *the* number for an options market maker. High gamma means your
delta hedge goes stale fast, so you must rebalance constantly, and every rebalance
costs the bid-ask spread. Gamma is largest **at the money and close to expiry**,
which is exactly when short-dated at-the-money options become dangerous to be
short. Measured here: an ATM option at T=0.05y has gamma 0.081 versus 0.017 at
T=1y — **4.7× more**.

**Vega** is largest for **long-dated at-the-money** options: volatility needs
*time* to matter. Same measurement: vega 8.9 at T=0.05y versus 38.2 at T=1y.
Long-dated options are the volatility instrument; short-dated options are the
gamma instrument.

**Theta and gamma are the same trade, mirrored.** Wherever gamma is large, theta
is large and negative. You are *paid* theta for being short gamma and you *pay*
theta for being long it. That trade-off is the fundamental economics of options,
and it's directly visible in the two panels of `results/greeks.png`.

### Units, where people get burned

Gamma and vega are **identical for a call and a put** on the same strike — put-call
parity differs by a forward, which has no gamma and no vega. Delta, theta and rho
are not. In this code, vega is per 1.00 of volatility (divide by 100 for "per vol
point") and theta is per year (divide by 365 for per day). Every desk uses a
different convention. Always state yours.

---

<a name="6"></a>
## 6. How you know your Greeks are right

A typo in a closed-form Greek is **invisible**: the number looks plausible and is
simply wrong. So `finite_difference_greeks` recomputes every Greek by bumping the
inputs and differencing the *price* — slow, but almost impossible to get wrong.

Central differences (`[f(x+h) − f(x−h)] / 2h`) are used because the error is
`O(h²)` rather than `O(h)`, and `h` is scaled to each variable's magnitude, since
a bump of 1e-4 means something very different for `S=100` than for `r=0.03`.

Across 36 configurations (3 spots × 3 maturities × 2 vols × call/put), worst
relative error: **2.06e-06**.

The test suite also asserts things that must be true structurally: gamma peaks at
the strike; `delta_call − delta_put = e^(−qT)` exactly; long-dated vega exceeds
short-dated by more than 3×; the price is strictly monotone in volatility and
strictly decreasing in strike.

---

<a name="7"></a>
## 7. Implied volatility: running the formula backwards

Volatility is the only Black–Scholes input nobody can observe — and the price is
on the screen. So invert: **what σ makes the model reproduce the market price?**

That number is the **implied volatility**. It is not a forecast and not the
"true" volatility. It is a **change of units** — a way of quoting a price that
strips out the mechanical effects of strike, spot and time so two very different
options can be compared. Bond traders quote in yield rather than dollars for
exactly the same reason.

**Why it's solvable.** The option price is *strictly increasing* in σ — more
volatility means more chance of a big favourable move, and the downside is
already capped at zero. Monotonicity means a unique root.

**Newton–Raphson.** The derivative of price with respect to σ is **vega**, which
is always positive:

```
σ_(n+1) = σ_n − (BS(σ_n) − market_price) / vega(σ_n)
```

Newton converges **quadratically** — the number of correct digits roughly doubles
each step. From the project output:

```
   iter       sigma    model price         error
  start  0.24083189    10.80354329     -1.25e+00
      1  0.27349471    12.05280395     -2.02e-04
      2  0.27350000    12.05300619     -1.46e-11
      3  0.27350000    12.05300619     -7.11e-15
```

`1e+00 → 1e-04 → 1e-11 → 1e-15`. Each error is roughly the square of the last.
Three steps to machine precision.

---

<a name="8"></a>
## 8. Where Newton breaks

Vega goes to zero for options deep in or deep out of the money. When vega is
tiny, the Newton step `−diff/vega` is enormous and the iterate flies off to a
negative or absurd volatility.

This is not hypothetical — it is the normal failure mode on the wings of a real
chain, which is exactly where you most want a good answer. Measured on a 2.5-week
chain in this project, vega spans **130 orders of magnitude** from at-the-money
to the far wing, and plain Newton fails at **46 of 70 strikes**.

The production approach is a hybrid, and this project implements all three parts:

**1. A good starting point.** Manaster–Koller (1982) give the initial guess that
*maximises vega* — the most numerically stable place to start:

```
σ₀ = sqrt( |ln(S/K) + rT| · 2/T )
```

**2. Newton with guards**, rejecting any step that leaves a sane bracket.

**3. Brent's method as a fallback.** Brent is *bracketing*: it keeps a lower and
an upper bound that straddle the root and never leaves them, so it cannot
diverge. Slower, but it always converges.

### The subtle part: converge in *volatility* space, not price space

The obvious stopping rule is "stop when `|model price − market price| < tol`".
**It is wrong, and it fails silently.**

Consider a deep OTM call worth 2e-9 with a vega of 2e-7. Any volatility within
about 1% of the truth reproduces that price to within 1e-8. So an absolute price
tolerance of 1e-8 declares victory while the volatility is still wrong **in its
second decimal place**.

This project had exactly that bug. It produced errors of **1.1e-2 in implied
vol** — over one whole vol point — on precisely the wing options where a surface
fit is most fragile.

The fix: the Newton step `diff/vega` *is* the remaining error in volatility, to
first order. So stop when the **step** is small, not when the residual is. Same
iteration, correct stopping rule, costs nothing. After the fix: 119 round trips,
worst error **4.5e-15**.

### When there is no answer at all

No-arbitrage bounds require `max(S·e^(−qT) − K·e^(−rT), 0) ≤ C ≤ S·e^(−qT)`. If
the quoted price falls outside that range, **no volatility reproduces it** and
the right answer is `NaN`, not a number.

This happens constantly with real data — stale quotes, wide spreads, mismatched
timestamps. Silently returning garbage instead of `NaN` is how bad volatility
surfaces get built.

---

<a name="9"></a>
## 9. The volatility smile

Black–Scholes says volatility is a property of the *stock*. If that were true,
every option on the same stock with the same expiry would imply the same
volatility, and a plot of implied vol against strike would be **flat**.

It is not flat. For equity indices it slopes steeply down: low strikes (puts)
imply much higher volatility than high strikes. That shape is the **skew** or
smirk. For FX it's closer to a symmetric smile.

The equity skew was mild before **October 1987** and has been steep ever since.
Two non-exclusive explanations:

**Crash risk.** Real returns have a fat *left* tail. The lognormal doesn't. The
market prices the extra tail risk by paying up for out-of-the-money puts, and
when you invert that price through a model that cannot represent fat tails, the
excess has to come out somewhere — it comes out as a higher implied volatility.

**Leverage / volatility feedback.** When equity falls, leverage rises and
volatility rises with it. Down moves and high volatility arrive together. This is
the *same effect* the GJR-GARCH `γ` parameter measures in §19, seen from the
options side. One phenomenon, two completely different datasets.

So the surface is not an embarrassment to be explained away. It is **the market's
entire risk-neutral distribution, written in Black–Scholes units**, and it is a
rich tradeable object.

---

<a name="10"></a>
## 10. SVI: parameterising the smile

Real chains are sparse, noisy and unevenly spaced. Spline through the points
naively and you get a surface that permits **arbitrage** — prices where somebody
can construct a guaranteed profit. Then every price you interpolate off it is
wrong, and any model you calibrate to it inherits the inconsistency.

**SVI** (Gatheral, 2004) is the industry standard: flexible enough to fit real
smiles, only five parameters, and its arbitrage conditions are checkable in
closed form.

Work in **total implied variance** `w(k) = σ²(k)·T` against **log-moneyness**
`k = log(K/F)`:

```
w(k) = a + b·( ρ·(k − m) + sqrt( (k − m)² + σ² ) )
```

| parameter | meaning |
|---|---|
| `a` | overall level — lifts the whole smile |
| `b ≥ 0` | slope of the wings; larger = steeper |
| `−1 < ρ < 1` | asymmetry. **Negative ρ tilts down to the right — that is the equity skew.** |
| `m` | horizontal shift — where the minimum sits |
| `σ > 0` | curvature at the bottom. Small = sharp V, large = smooth bowl. |

**Why total variance rather than volatility?** Two reasons. Total variance is the
quantity that must be monotone in time for no calendar arbitrage. And the SVI
wings are asymptotically **linear in `k`** in total-variance space — which is
exactly what Roger Lee's moment formula says any arbitrage-free smile must do.
The parameterisation is built to respect that automatically.

### Fitting it

Three practical things matter more than the optimiser you pick.

**Weight by vega.** An at-the-money option's price is very sensitive to
volatility, so its implied vol is well determined. A far OTM option has almost no
vega, so its implied vol is a wildly noisy number inferred from a price that is
mostly bid-ask spread. Fitting them equally lets the least reliable points drag
the curve around.

**Seed cleverly.** The SVI objective is non-convex with local minima that look
fine and fit badly. But notice: with `y₁ = k − m` and `y₂ = sqrt((k−m)² + σ²)`,

```
w = a + (b·ρ)·y₁ + b·y₂
```

For *fixed* `(m, σ)` that's an ordinary **linear regression** in `(a, b·ρ, b)`,
solvable exactly. So the genuinely non-convex part is only **two-dimensional**.
This project sweeps a grid of `(m, σ)`, solves the inner linear problem at each
node, and starts the non-linear fit from the best few. (This is the idea behind
Zeliade's "quasi-explicit" SVI calibration.)

**Floor σ.** As `σ → 0` the smile degenerates into a sharp V with a kink — which
can fit a handful of noisy points beautifully and is complete nonsense as a
volatility smile (it implies an atom of probability at one strike). It is an
attractive local optimum. Without a hard floor, one slice in the test chain
collapsed into exactly that: `σ = 0.0000` and an RMSE of **7.6 vol points** while
every other slice fitted to 0.1.

### Does it work?

The synthetic chain is generated from a *known* SVI surface, so the fit can be
scored against the truth. Errors in volatility points:

| T (years) | usable quotes | max err | RMS err |
|---|---|---|---|
| 0.019 | 10 | 3.202 | 1.542 |
| 0.083 | 18 | 0.820 | 0.270 |
| 0.250 | 23 | 0.132 | 0.087 |
| 0.500 | 24 | 0.335 | 0.205 |
| 1.000 | 25 | 0.143 | 0.088 |
| 2.000 | 25 | 0.259 | 0.124 |
| **average** | | | **0.386** |

For context, a typical bid-ask spread on a liquid option is worth well over half
a vol point, so sub-0.5 recovery is at the noise floor of the data.

The shortest expiry is always worst, and that is **not a bug**. A one-week option
has almost no time value in the wings, so most of its strikes get filtered out as
sub-tick, leaving very few usable quotes to pin down five parameters. Real desks
have the same problem and handle it the same way: fewer parameters, or borrowing
shape from the neighbouring expiry.

---

<a name="11"></a>
## 11. Arbitrage: the two conditions that must hold

### Butterfly (within one expiry)

A butterfly spread — buy `K−dK`, sell two `K`, buy `K+dK` — has a payoff that is
never negative. So it must never cost less than nothing. Equivalently, the
implied risk-neutral **density must be non-negative everywhere**.

**Durrleman's condition** expresses this directly on the smile:

```
g(k) = (1 − k·w'/(2w))² − (w'²/4)·(1/w + 1/4) + w''/2  ≥  0
```

If `g(k) < 0` anywhere, your surface implies a **negative probability** at that
strike. Everything downstream is then meaningless.

### Calendar (across expiries)

Total implied variance must be **non-decreasing in maturity** at every fixed
log-moneyness:

```
w(k, T₁) ≤ w(k, T₂)   for T₁ < T₂
```

Otherwise a calendar spread is free money: variance cannot un-accumulate.

Both are checked and reported rather than assumed. The project's fitted surface
passes both, with `min g` around +0.23 across all slices.

**And the checker is itself tested**: feeding it a deliberately arbitrageable
smile (`a=−0.02, b=0.9, ρ=−0.99, σ=0.01`) must produce a violation. A test that
never fails is not a test.

---

<a name="12"></a>
## 12. Cleaning an option chain

This step decides the quality of everything after it. Each filter removes a
specific real pathology:

| filter | what it's catching |
|---|---|
| crossed quote (bid > ask) | broken data |
| **mid outside [bid, ask]** | a stale mid next to a live bid/ask |
| zero bid | nobody is willing to buy at any price |
| price below $0.05 | a $0.02 option is one tick wide; its IV moves several vol *points* if the quote moves one cent |
| relative spread > 60% | the "price" is a fiction; the true value is somewhere in a huge interval |
| no open interest | a contract nobody holds has a quote nobody keeps accurate |
| extreme moneyness | beyond ±1 log-moneyness the option is worth pennies and its IV is noise |
| under 3 days to expiry | time value collapses, pin risk dominates, IV is unstable |
| in-the-money | see below |

**The OTM-only convention.** For in-the-money options most of the price is
intrinsic value, so the IV is inferred from a small residual and is
correspondingly noisy. It also sidesteps early-exercise effects for American
options. And you lose nothing: by put-call parity the OTM put and the ITM call
carry the same information, so you keep the liquid one.

**Always look at the removal counts.** If one filter is eating most of your data,
either the filter or the feed is wrong and you need to know which.

The `mid outside [bid, ask]` check deserves special mention because it sounds too
obvious to be worth writing. It caught a stale at-the-money quote implying **71%
vol against a true 31%** — invisible to every other filter, because the price on
its own still looked perfectly plausible. One such quote threw an entire maturity
slice off by **7 volatility points**, because vega weighting gives at-the-money
quotes the *most* influence.

### Robust fitting

Even after cleaning, real chains contain points that survive every mechanical
filter and are still wrong. Ordinary least squares has a **breakdown point of
zero**: one arbitrarily bad point moves the fit arbitrarily far.

So: fit once, measure each point's residual, discard anything beyond 4 robust
standard deviations, refit. Crucially, scale by the **median absolute deviation**
(`1.4826 × MAD` estimates σ for Gaussian data) rather than the standard
deviation — because the standard deviation is itself inflated by the outliers you
are trying to find.

---

<a name="12b"></a>
## 12b. Real chains: extracting the forward and rate

Every example so far used the risk-free rate `r` and dividend yield `q` as
known inputs. On a real chain they are not known — they have to be pulled
out of the quotes themselves, and doing that correctly turned out to matter
a lot more than it sounds. This section is the SPY/AAPL/TSLA calibration in
`results/RESULTS.md` from the inside.

**The identity.** For a call and a put at the *same* strike `K` and expiry
`T`:

    C − P = D(F − K)          D = e^{−rT}   the discount factor
                               F             the forward price of the stock

This is **model-free** — no Black-Scholes, no volatility, no distributional
assumption. It's true because a long call plus a short put has exactly the
payoff of a forward contract, in any model. Rearranged, `C − P` is linear in
`K`, with slope `−D` and intercept `D·F`.

**The first attempt, and why it failed.** The obvious plan is to fit both
unknowns — `D` (hence `r`) and `F` — from that one line, using a weighted
regression across every strike with a liquid call and put. Run on the real
SPY chain, this produced per-expiry `r` estimates of −23%, +14%, −21%, −5%,
−23%... swinging wildly between *adjacent weekly expiries*, while the
forward `F` the same regression produced stayed put at 762–764 (spot was
761.69) the entire time.

That is not five different rates. It's a signal-to-noise problem. At
`T = 5` days, `D = e^{-0.04 × 5/365} ≈ 0.99945` — the discount factor is
within a few hundredths of a percent of 1, so the regression has to detect
a rate signal smaller than the bid-ask-driven pricing noise already in the
quotes. `r` (equivalently `D`) is simply **not statistically identifiable**
from a handful of days' worth of option prices — there isn't enough time
value of money in five days for the market to reveal what `r` is. `F`
doesn't have this problem: it's pinned by *where the line crosses zero*, a
geometric feature of the data that stays well-determined however noisy the
line's exact slope is.

**The fix**, used throughout `results/RESULTS.md`: fix `r` at a known,
externally observed rate (this project uses 4.2% throughout, matching
`generate_synthetic_chain`'s default), and solve for only the one quantity
the chain can actually reveal — `F`. With `D` now fixed and known, every
liquid strike gives an independent estimate `Fᵢ = (Cᵢ − Pᵢ)/D + Kᵢ`; the
per-expiry forward is their spread-weighted average, and the dividend/borrow
yield falls out as `q = r − ln(F/S)/T`. This mirrors what a real desk does:
assume a known short-term risk-free/SOFR rate (public, and nearly the same
for every US equity on a given day), and extract the thing that genuinely
isn't observable — for a single name, `q` mixes the trailing dividend yield
with the stock's *short-sell borrow cost*, which is real, time-varying, and
not published anywhere.

**Why misspecifying `r` barely matters, which is exactly why it's
unidentifiable.** The Black-Scholes formula is
`e^{-rT}[F·N(d1) − K·N(d2)]`, where `d1, d2` depend on `ln(F/K)`, `σ`, and
`T` — the *shape* of the smile is a function of the forward alone, not of
how `r` and `q` individually combine to produce it. Only the outer discount
factor `e^{-rT}` depends on `r` directly, and at the short maturities where
`r` is hardest to pin down, that factor is within a few basis points of 1
for any economically reasonable `r` anyway. Getting `r` wrong by a whole
percentage point moves a short-dated price by a few hundredths of a
percent — which is *also* the reason it can't be measured from the price in
the first place. The same fact makes it safe to assume and impossible to
extract; that isn't a coincidence.

**A second-order effect this produces, worth knowing before it surprises
you**: even with `r` fixed, `q = r − ln(F/S)/T` still divides by `T`, so
the *short-dated* `q` estimates are still noisy — SPY's extracted `q` reads
−8.8% on the 6-day expiry, improving to +0.5% by the one-year expiry (see
`results/RESULTS.md` §2). A −8.8% "dividend yield" is not a claim that SPY
is about to pay a huge dividend; it's a few cents of genuine bid-ask noise
in `F`, annualised over six days. Trust the long end of a `q` curve far
more than the short end, for the identical reason you can't trust a
short-dated `r`.

---

<a name="13"></a>
## 13. Monte Carlo

**Why simulate when a formula exists?** Two reasons.

**Validation.** You know the right answer, so you can check that the simulation
converges to it, that the error shrinks at the rate theory predicts, and that
your variance-reduction tricks are *correct* rather than merely plausible.
Building a Monte Carlo engine against a known answer first is how you earn the
right to trust it on a problem with no known answer.

**Most real payoffs have no closed form.** Asians, barriers, lookbacks,
autocallables, anything path-dependent or multi-asset. Monte Carlo handles them
all with the same code.

**The method.** Under the risk-neutral measure the exact solution of the GBM SDE
is

```
S_T = S₀·exp( (r − q − σ²/2)·T + σ√T·Z ),    Z ~ N(0,1)
```

This is **exact, not a discretisation**. For a payoff depending only on `S_T`
there is no time-stepping error at all — one normal draw per path, and the only
error is statistical. (Path-dependent payoffs do need time steps, which is why
the Asian pricer takes `n_steps`.)

Then `price = e^(−rT) × average of payoff(S_T)`.

**The central problem: the √N wall.** The standard error is `s/√N`. That square
root is brutal: **ten times more accuracy costs one hundred times more paths**.

You cannot beat the `√N` *rate* with plain Monte Carlo. What you *can* do is
shrink the numerator `s`. That is what variance reduction is, and it is the whole
craft.

Verified in this project — log-log slope of standard error against N:

```
plain                 -0.4994
antithetic            -0.4991
control               -0.4966
antithetic+control    -0.4985
                theory: -0.5000
```

A different slope would mean the estimator is not a sample mean of i.i.d. draws —
which would be a bug.

---

<a name="14"></a>
## 14. Variance reduction

### Antithetic variates

For every draw `Z`, also use `−Z`. Both are valid N(0,1) draws, so the estimator
stays unbiased. The pair is negatively correlated: when `Z` gives a high stock
price, `−Z` gives a low one, and their errors partially cancel.

```
Var( (X + X')/2 ) = ( Var(X) + Var(X') + 2·Cov(X, X') ) / 4
```

With `Cov < 0` this beats two independent draws.

The gain for a call is **modest**, and that is expected: `max(S_T − K, 0)` is
**convex** in `Z`, and convexity partly undoes the cancellation. Antithetics
shine on near-linear payoffs, and can even *hurt* for strongly convex ones —
worth knowing.

### Control variates — the big one

Alongside the thing you want, compute something correlated whose **exact mean you
already know**. Here: `S_T` itself, with `E[e^(−rT)·S_T] = S·e^(−qT)`.

On each path you can *see* the error the simulation made on the control — and
because the payoff is correlated with the control, you can infer part of the
error it made on the payoff and subtract it:

```
estimator = mean( payoff + c·(S_T − E[S_T]) )
```

Still unbiased, because the correction has mean zero by construction. The optimal
coefficient is `c* = −Cov(payoff, control)/Var(control)`, and it delivers

```
Var_reduced = Var_original × (1 − ρ²)
```

**That formula is the one to remember.** ρ = 0.9 removes 81% of the variance;
ρ = 0.99 removes 98%.

And it is testable, which is the point. Measured in this project:

```
control variate: rho = 0.8779 -> theory predicts 4.36x, observed 4.36x
```

The observed reduction matching `1/(1−ρ²)` to two decimal places is the check
that the implementation is *right*, not merely helpful. That's a much stronger
test than "the variance got smaller".

---

<a name="15"></a>
## 15. The payoff with no formula

An **arithmetic-average Asian call**: `payoff = max( average(S_t) − K, 0 )` over
52 weekly observations.

There is no Black–Scholes formula, because a **sum of lognormals is not
lognormal**.

But a sum of **normals** *is* normal — so the **geometric**-average Asian, whose
average is `exp(mean of the logs)`, is exactly lognormal and **does** have a
closed form. That makes it a near-perfect control variate for the arithmetic one:
same paths, correlation above 0.999, and one of the two answers known exactly.

Measured:

```
geometric Asian call (exact)        4.031841
arithmetic Asian, plain MC          4.270873  +/- 0.018183
arithmetic Asian, controlled MC     4.285546  +/- 0.000702

correlation with the control        0.999262
variance reduction                       671x
```

You would need **671 times** more plain paths for the same accuracy.

Three sanity checks that must hold, and do:

- Asian (4.29) < vanilla (8.96) — averaging reduces effective volatility
- arithmetic (4.29) ≥ geometric (4.03) — the AM–GM inequality, path by path
- the controlled and plain estimates agree within their error bars

Asians are used in commodities and FX precisely because averaging makes the
payoff harder to manipulate near expiry.

---

<a name="16"></a>
## 16. Greeks from Monte Carlo

**The naive way**: rerun the simulation at `S` and at `S+h`, and difference. Two
problems. It costs two simulations. And if you use *different* random numbers for
the two runs, the Monte Carlo noise swamps the small real difference — your
"delta" is mostly noise. Using the **same** random numbers (common random
numbers) fixes most of that.

**The pathwise way**: differentiate *inside* the expectation. Since
`S_T = S·exp(...)`, we have `dS_T/dS = S_T/S`, so by the chain rule for a call:

```
delta = e^(−rT)·E[ 1{S_T > K} · S_T/S ]
```

One simulation, no bump, no differencing noise, and it reuses paths you already
have. Measured: pathwise 0.512272 ± 0.000869 against an analytic 0.511587 —
**0.79 standard errors**.

**The catch**: this needs the payoff to be differentiable almost everywhere. Fine
for calls and puts. **Not valid** for a digital option, whose payoff is a step
function — there you need the likelihood-ratio method or a smoothed payoff.
Knowing which method applies to which payoff is a real interview question.

---

<a name="17"></a>
## 17. Volatility clustering and GARCH

Look at any real return series and volatility is obviously not constant. What you
see is **volatility clustering**: quiet weeks follow quiet weeks, and once things
get violent they stay violent. Mandelbrot noticed this in 1963.

Concretely, from this project's data:

```
   lag    corr(r_t, r_(t−k))    corr(r²_t, r²_(t−k))
     1              -0.0147                  0.2374
     2               0.0126                  0.2181
     5              -0.0094                  0.2433
    10               0.0170                  0.2470
    22              -0.0328                  0.1693
```

**That table is the entire argument for GARCH:**

1. **Returns are nearly unpredictable.** Left column ≈ 0 at every lag. You cannot
   forecast direction.
2. **Squared returns are strongly predictable.** Right column clearly positive
   and decaying slowly. Today's move tells you nothing about tomorrow's
   *direction* but a great deal about tomorrow's *magnitude*.
3. **Fat tails.** Excess kurtosis +1.20 (a normal has exactly 0).

### The model

```
r_t       = μ + ε_t
ε_t       = σ_t·z_t,          z_t ~ iid, mean 0, variance 1
σ²_t      = ω + α·ε²_(t−1) + β·σ²_(t−1)
```

Read the variance equation as a weighted average of three things:

- `ω` pulls toward the long-run level
- `α·ε²_(t−1)` — **yesterday's surprise**. Big move yesterday → raise today's
  forecast. `α` is how reactive you are.
- `β·σ²_(t−1)` — **yesterday's forecast**. This is the memory: it's what makes
  shocks decay slowly rather than instantly.

**Constraints**: `ω > 0`, `α, β ≥ 0`, and `α + β < 1` for stationarity.

That last one carries all the meaning. **`α + β` is the persistence.** The
long-run variance is `ω/(1 − α − β)`, which only exists if `α + β < 1`. A shock
decays like `(α+β)^h`, so the **half-life** of a volatility shock is
`ln(0.5)/ln(α+β)` days.

For daily equity data you almost always get `α ≈ 0.05–0.12`, `β ≈ 0.85–0.93`,
persistence ≈ 0.97–0.99. This project's fit: **α = 0.084, β = 0.900, persistence
= 0.984, half-life = 44 days**.

That is why constant-volatility Black–Scholes misprices options: today's
volatility is highly informative about the next month's.

### GARCH generates fat tails for free

Note that GARCH produces excess kurtosis **even with normal innovations**,
because it is a mixture of normals with time-varying variance. So fat tails are
partly *explained* by the model rather than needing to be assumed.

---

<a name="18"></a>
## 18. Estimating GARCH by maximum likelihood

Given parameters, the conditional variance path is fully determined by recursion.
Under normal innovations:

```
LL = −0.5 · Σ_t [ log(2π) + log(σ²_t) + ε²_t/σ²_t ]
```

Maximise numerically. Three implementation details matter more than the choice of
optimiser.

**1. Parameter transforms instead of constrained optimisation.** Optimise over
unconstrained reals and map them into the valid region:

```
ω = exp(u₀)
α = 0.999·sigmoid(u₁)
β = 0.999·(1 − α)·sigmoid(u₂)
```

Every constraint is satisfied **by construction** — the optimiser can never
propose a negative variance or a non-stationary model, and never has to crawl
along a boundary. Much more robust than penalties.

**2. Scaling.** Daily returns are ~0.01, so their squares are ~1e-4 and ω is
~1e-6. Optimisers work in absolute step sizes and quietly do badly on numbers
that small. Fitting in *percent* (×100) and converting back is one line, and it's
the difference between reliable convergence and mysterious failures. Nobody tells
you this.

**3. The recursion is a linear filter.** `σ²_t = [ω + shock(ε_(t−1))] + β·σ²_(t−1)`.
The bracket depends only on `ε`, which is known up front — so the whole thing is a
first-order IIR filter, exactly `scipy.signal.lfilter([1], [1, −β], x)`.

This matters because MLE calls it thousands of times. The naive Python loop made
one fit take **7.28 s**; the filter version takes **0.20 s** — a 36× speedup with
**bit-identical** output (asserted in the tests). Without it, the out-of-sample
rolling study — hundreds of refits — would be impossible.

### Standard errors

MLE theory: the estimator is asymptotically normal with covariance equal to the
inverse **observed information matrix** (the Hessian of the negative
log-likelihood at the optimum). We difference the Hessian numerically in the
unconstrained parameters, then push through to the constrained ones with the
**delta method**: `Var(g(u)) ≈ J·Var(u)·J'`.

Standard errors are what turn "γ = 0.09" into "γ = 0.09 with a t-statistic of
5.6, so the leverage effect is real and not noise". **Reporting a point estimate
with no uncertainty is the clearest sign someone hasn't actually done
statistics.**

### The validation that matters

Fit to *real* data and you have no idea whether the answer is right — the truth
is unobservable. Fit to data you **simulated from known parameters** and you can
check that it recovers them:

```
  param   true          estimated      z
  omega   2.0e-06       2.26e-06     +0.56
  alpha   0.09          0.0933       +0.35
  beta    0.89          0.8843       -0.49
  mu      0.0004        0.000416     +0.13
```

Every z well inside ±2. If an estimator cannot recover parameters from data
generated by its own model, nothing it says about real data means anything.

---

<a name="19"></a>
## 19. The leverage effect

Plain GARCH treats +3% and −3% identically, because it only sees `ε²`. Equity
markets do not work that way: volatility rises far more after a fall than after a
rally of the same size.

**GJR-GARCH** (Glosten–Jagannathan–Runkle) adds one term:

```
σ²_t = ω + (α + γ·1{ε_(t−1) < 0})·ε²_(t−1) + β·σ²_(t−1)
```

A significantly positive `γ` is the leverage effect. This project's fit:
**γ = 0.081, t = 5.64** — and GJR beats plain GARCH on both AIC and BIC.

The test suite also checks the **negative** case: fitted to *symmetric* simulated
data, GJR returns γ = 0.0096 with |t| = 0.73. It does not invent a leverage
effect that isn't there.

**This is the statistical twin of the volatility skew in §9.** Out-of-the-money
puts are expensive precisely because down moves come with more volatility. One
phenomenon, two completely different datasets, two completely different
methodologies — and they agree. That connection is one of the more satisfying
things in this project.

The **news impact curve** (Engle–Ng, 1993) makes it visible: plot today's
variance against yesterday's shock. GARCH gives a symmetric parabola; GJR gives a
parabola with a kink at zero and a steeper left arm. See
`results/news_impact_curve.png`.

### Model selection

```
GARCH normal     AIC 9269.7   BIC 9294.4
GJR   normal     AIC 9240.7   BIC 9271.5   <- best
GJR   Student-t  AIC 9245.1   BIC 9282.0
```

AIC and BIC both penalise extra parameters; BIC penalises harder. Adding a
parameter *always* improves the raw likelihood, so comparing likelihoods alone
would always pick the biggest model. Here the Student-t adds a parameter and
estimates ν ≈ 150 (essentially normal), so the criteria correctly reject it — the
data was simulated with normal innovations, and model selection found that out.

---

<a name="20"></a>
## 20. Forecasting, and how to evaluate it honestly

### Multi-step forecasts

For `h > 1` we don't know the future shock, but its *expected* square is the
variance itself, so the recursion collapses to:

```
E[σ²_(t+h)] = LR + persistence^(h−1) · (σ²_(t+1) − LR)
```

Geometric decay from today's level back toward the long-run level `LR`. With
persistence 0.98 that decay is slow — after a month you're still nowhere near the
long run — which is exactly what makes GARCH useful for pricing options a month
or more out.

For option pricing you want the **average** variance over the option's life, not
the instantaneous value at expiry. Black–Scholes wants the average.

### The rule that decides whether any of this is real

At each point in time the model may only use data available **up to that point**.
This project refits on an expanding window every 25 days and forecasts forward.

Fitting once on the whole sample and then "forecasting" the same data is
in-sample fitting. It always looks excellent and it means nothing.

### The competitors

**EWMA (RiskMetrics)**: `σ²_t = λ·σ²_(t−1) + (1−λ)·r²_(t−1)`, λ = 0.94.

This is GARCH(1,1) with **ω = 0** and persistence **exactly 1**. That has a real
consequence: with no ω there is no long-run level to revert to, so the multi-step
forecast is **flat forever**. EWMA is a fine one-day model and a poor one-month
model, and knowing why is the point of including it.

**Constant**: trailing 252-day sample variance. The honest null hypothesis. If
you can't beat this, you've achieved nothing.

### Scoring, and why RMSE alone is not enough

The "true" volatility is never observed. You compare against a **proxy** — usually
the squared return — and the squared return is an extremely noisy proxy: unbiased
for the variance, but with enormous variance of its own (for a normal,
`Var(r²) = 2σ⁴`).

Patton (2011) showed only certain loss functions still rank forecasts correctly
when the target is a noisy but unbiased proxy. MSE is one. **QLIKE** is the
other, and it's generally preferred because it's far less dominated by a handful
of huge days:

```
QLIKE = mean( proxy/forecast − log(proxy/forecast) − 1 )
```

Zero when forecast equals proxy, positive otherwise, and it penalises
**under**-forecasting more heavily than over-forecasting. That asymmetry is the
right one for risk management: being caught short of volatility is worse than
carrying too much.

### Results

2,500 out-of-sample days:

| model | RMSE (vol pts) | MAE | QLIKE | mean forecast |
|---|---|---|---|---|
| **GARCH** | **10.174** | 8.210 | **1.23234** | 15.47% |
| EWMA(0.94) | 10.204 | 8.171 | 1.24410 | 15.27% |
| constant | 11.161 | 9.044 | 1.39009 | 15.84% |

GARCH beats constant volatility by **8.8% on RMSE and 11.3% on QLIKE**.

**A word on magnitudes.** The improvement looks modest, and that is *not* a
weakness of GARCH — it is the noise in the proxy. Most of that RMSE is
irreducible measurement error that no forecast could ever remove. **Compare
models against each other, never against zero.** QLIKE, being less sensitive to
that noise, shows the gap more clearly.

---

<a name="21"></a>
## 21. Implied vs realised: the variance risk premium

This is where the two halves of the project meet.

```
GARCH 22-day forecast (real-world measure P)     25.85%
At-the-money implied volatility (risk-neutral Q) 26.85%
Gap                                              +1.00 vol points
```

**These two numbers are not supposed to be equal**, and understanding why is one
of the most useful things here.

GARCH estimates volatility under the **real-world measure P**: what will actually
happen, on average. Implied volatility lives under the **risk-neutral measure Q**:
what the market charges. The difference is the **variance risk premium**, and it
is one of the most robust findings in empirical finance — implied volatility
exceeds subsequent realised volatility the large majority of the time, across
markets and across decades.

**Why would anyone pay it?** Because volatility spikes exactly when everything
else in your portfolio is falling. An asset that pays off in a crash is worth
more than its expected payoff, so investors accept a negative expected return to
hold it — and option sellers are paid for supplying it. The premium is
compensation for a genuinely nasty risk profile: small steady gains punctuated by
large losses.

This gap is the basis of the short-variance trade, and of why that trade
periodically destroys the people running it.

---

<a name="22"></a>
## 22. The bugs I hit, and what each one taught

Worth more in an interview than the features.

### Bug 1: converging on the wrong quantity

Covered in §8. The implied-vol solver stopped on an absolute *price* tolerance,
which on a low-vega wing option meant declaring success while the volatility was
still wrong in its second decimal — **1.1e-2 of silent error**. Fixing the
criterion to measure the Newton step (i.e. convergence in volatility space) took
the worst round-trip error to **4.5e-15**.

**Lesson**: converge in the quantity you actually want, not the one that happens
to be in front of you. And a tolerance without units attached is a bug waiting to
happen.

### Bug 2: the slice that collapsed into a V

One SVI slice fitted with `σ = 0.0000` and an RMSE of 7.6 vol points while every
other slice fitted to 0.1. The degenerate `σ → 0` V-shape is an attractive local
optimum that fits noisy points beautifully and is meaningless as a smile.

Two fixes: a hard **floor** on σ, and much better seeding via the linear-in-three-
parameters structure (§10).

**Lesson**: in a non-convex fit, "it converged" tells you nothing. Look at the
parameters and ask whether they are physically sensible.

### Bug 3: one stale quote, seven vol points

The slice above had a *second* problem: a stale at-the-money mid implying 71% vol
against a true 31%. It passed every filter I had, because the price on its own
looked plausible — but its mid sat **outside its own bid/ask**.

Two fixes: the `mid outside [bid, ask]` filter, and robust MAD-based outlier
rejection with a refit.

**Lesson**: vega weighting means at-the-money errors are the *most* damaging.
Data cleaning is not the boring part of the pipeline; it is the part that decides
whether the pipeline works.

### Bug 4: seven seconds per fit

The GARCH recursion as a Python loop made each fit take 7.28 s, which made the
out-of-sample study (hundreds of refits) impossible. Recognising it as a
first-order IIR filter and using `scipy.signal.lfilter` gave a **36× speedup with
bit-identical output**.

**Lesson**: before reaching for a compiler or a JIT, ask whether the recursion is
a *linear filter*. A surprising number of them are, and the vectorised primitive
is already in scipy.

### Bug 5: thresholds that tested my guess, not the theory

My first Monte Carlo test asserted "the control variate cuts variance by at least
5×". It failed at 4.4×. My threshold was wrong, not the code — the correlation
was 0.878, and `1/(1 − 0.878²) = 4.36`, exactly what was observed.

So I replaced the arbitrary threshold with a test that the observed reduction
matches `1/(1 − ρ²)` to within 15%.

**Lesson**: a test that asserts a number you guessed is weak. A test that asserts
a *theoretical relationship* is strong — it validates the mathematics rather than
your expectations, and it fails for the right reasons.

### Bug 6: a curve that fits perfectly and extrapolates to nonsense

Calibrating SPY's real chain, one slice (13 days to expiry) fit with
`rho = +0.994` — the textbook reading of a positive `rho` is call-side
skew, the opposite of what every equity index shows, and the opposite of
what that exact slice's raw quotes plainly showed by eye (74% vol on the
deepest put, 14% near the money, a normal-looking put skew). The fit
wasn't lying about its own data: RMSE was 0.14 vol points, excellent. The
observed strikes for that expiry only spanned `k` from −0.25 to +0.05 — a
narrow, one-sided window — and the optimizer had put the SVI curve's vertex
at `m = +0.61`, nowhere near any strike anyone had actually quoted. Inside
the observed window, the fitted curve was still decreasing in `k` (correct
shape) purely because the whole window sat to the left of the vertex; `rho`
had become free to take whatever sign made that possible, because nothing
in the data disagreed with it.

The same slice, fed into the calendar-arbitrage check (which compares
neighbouring expiries at a shared, fixed range of `k`), produced a total-
variance violation of magnitude **10.5** — not a rounding-error violation,
a "this is not a volatility surface" violation. The checker was comparing
this slice's wild extrapolation against a well-behaved neighbour, at `k`
values this slice had never seen a quote for.

Two separate fixes, not one:

1. `SVIParams` now records the log-moneyness range it was actually fit on
   (`k_lo`, `k_hi`), and `SVIParams.reliable()` reports whether the fitted
   vertex falls inside that range (plus a small tolerance). A slice can have
   an excellent RMSE and still be flagged unreliable — those are answers to
   different questions.
2. `build_surface`'s calendar check now uses the *intersection* of every
   slice's observed range, not a fixed default. The same SPY violation
   dropped from 10.5 to 5.7 × 10⁻³ — still real, but now two orders of
   magnitude smaller and traceable to a specific, correctly-flagged slice,
   rather than to the checker asking a curve about strikes it never had.

**Lesson**: a low RMSE only tells you the fit matches the data it was given.
It says nothing about the fit anywhere *else* — and every parameter you read
off a fitted curve (like `rho`'s sign, conventionally "the skew") is only
trustworthy inside the region the data actually constrained. This is exactly
the failure mode the synthetic validation chain, generated with strikes
symmetric around the money at every maturity, structurally cannot produce —
real chains have uneven, sometimes one-sided strike coverage per expiry, and
a validation suite that never exercises that case will never catch it. See
`docs/CONCEPTS.md` §12b and `results/RESULTS.md` §4 for the numbers.

---

## Where to go next

- `docs/INTERVIEW.md` — the questions, and how to answer them.
- The code. Every module opens with a long comment written for someone who has
  read this.
- Extensions worth doing, roughly in order of value: local volatility via the
  Dupire equation; a Heston stochastic-volatility model calibrated to the same
  surface; American options by binomial tree or Longstaff–Schwartz;
  delta-hedging simulation to measure realised P&L against theoretical price; and
  a HAR-RV model using realised variance from intraday data, which usually beats
  GARCH when you have the data for it.
