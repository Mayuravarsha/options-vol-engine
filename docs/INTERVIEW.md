# Interview preparation: the options and volatility project

Read `CONCEPTS.md` first. This is the compressed Q&A form.

Two rules:

1. **Lead with the validation, not the feature.** "I built a Black-Scholes
   pricer" is nothing. "I validated every Greek against finite differences to
   2e-6, and the Monte Carlo variance reduction matches `1/(1-rho^2)` to two
   decimals" is a different conversation.
2. **Have a bug story ready.** Everyone's pricer "works". Very few candidates can
   describe a subtle numerical failure and what it taught them.

---

## The 60-second version

> I built an options pricing and volatility engine in Python. Black-Scholes with
> all the Greeks, validated against finite differences to about 2e-6, and
> put-call parity exact to machine precision. Implied vol by Newton-Raphson with
> a Manaster-Koller start and a Brent bracketing fallback — the interesting part
> there was that I originally converged on a price tolerance, which silently
> gives you volatility errors of a whole vol point on low-vega wing options, so I
> switched to converging in volatility space.
>
> On top of that, an SVI volatility surface with butterfly and calendar
> arbitrage checks, fitted on a synthetic chain generated from a known surface so
> I can score the recovery — it comes back to about 0.39 vol points on average.
> A Monte Carlo engine where the control-variate variance reduction matches the
> theoretical `1/(1-rho^2)` exactly, which is how I know it's right rather than
> just helpful. And GARCH and GJR-GARCH written from scratch by maximum
> likelihood, validated by recovering known parameters from simulated data, with
> a genuinely out-of-sample rolling forecast comparison against EWMA and constant
> volatility.
>
> The thing that ties it together is the last chart: my GARCH forecast sits about
> a vol point below at-the-money implied volatility. That gap is the variance
> risk premium — real-world measure versus risk-neutral measure — and it's the
> same phenomenon as the leverage effect I estimate in GJR, seen from the options
> side instead of the returns side.

---

## Black-Scholes

**Q: Derive Black-Scholes, or explain the idea.**

You don't need the PDE. The idea is replication: sell a call and buy delta
shares, choosing delta so the portfolio is instantaneously insensitive to the
stock price. That portfolio is riskless, so it must earn the risk-free rate, or
there's an arbitrage. That single requirement pins the price down uniquely.

Two consequences. The stock's expected return doesn't appear — the hedge removes
directional exposure, so only volatility matters. And pricing becomes an
expectation under the risk-neutral measure, a fictional world where everything
drifts at `r`. That's a computational device, not a claim about reality.

**Q: What is N(d2)?**

The risk-neutral probability the option finishes in the money. So `K·e^(-rT)·N(d2)`
is what you expect to pay, discounted. `N(d1)` is not a probability on its own —
it's a probability re-weighted by the stock's value in each state, so
`S·e^(-qT)·N(d1)` is the discounted expected value of what you receive. The
formula reads as "what you get minus what you pay", each weighted by how likely
it is.

**Q: Which Black-Scholes assumption is most wrong?**

Constant volatility, and it fails in a measurable way. Real returns have fat
tails and volatility clusters. The market knows, so it charges more for
out-of-the-money options than constant-vol Black-Scholes says. Invert those
prices and you get the volatility smile instead of a flat line. The whole
surface half of my project is about modelling that failure.

**Q: What's the first thing you'd check in someone's option pricer?**

Put-call parity. `C - P = S·e^(-qT) - K·e^(-rT)` is pure no-arbitrage — it holds
in *any* model, because a call minus a put replicates a forward. If the residual
isn't zero to machine precision, the code is wrong, not the model. Mine is 0.0
exactly.

On real data the gap isn't zero, and its size is informative — it absorbs the
spread, a wrong dividend, a wrong rate, and the early-exercise premium. Traders
back the implied dividend or borrow cost out of exactly that number.

---

## Greeks

**Q: Explain gamma to someone non-technical.**

Delta tells you how many shares to hold to hedge. Gamma tells you how fast that
number goes stale. High gamma means you're rebalancing constantly, and every
rebalance costs the spread. It's largest at the money and close to expiry —
which is exactly when short-dated at-the-money options become dangerous to be
short.

**Q: Relationship between gamma and theta?**

Two sides of the same trade. Wherever gamma is large, theta is large and
negative. You're paid theta for being short gamma and you pay theta for being
long it. If you're long gamma you make money re-hedging a moving market, and you
pay for that privilege in time decay.

**Q: Which options have the most vega?**

Long-dated at-the-money. Volatility needs time to matter. In my numbers, an ATM
option has vega 8.9 at T=0.05y versus 38.2 at T=1y. Long-dated options are the
volatility instrument; short-dated ones are the gamma instrument.

**Q: Are gamma and vega the same for a call and a put?**

Yes, identical, same strike. Put-call parity differs by a forward, which has no
gamma and no vega. Delta, theta and rho are not — `delta_call - delta_put =
e^(-qT)` exactly. I assert all of that in the test suite.

**Q: How do you know your Greeks are right?**

Finite differences. A typo in a closed-form Greek is invisible — the number looks
plausible and is wrong. So I recompute every Greek by bumping inputs and
differencing the price, using central differences for `O(h²)` error and scaling
`h` to each variable's magnitude. Worst relative error across 36 configurations:
2.06e-06.

---

## Implied volatility

**Q: What is implied volatility, really?**

A change of units. It's not a forecast and not the true volatility — it's the
volatility that makes Black-Scholes reproduce the market price. Traders quote in
vol for the same reason bond traders quote in yield: it strips out the mechanical
effects of strike, spot and time so different options become comparable.

**Q: Why can you invert it at all?**

The price is strictly increasing in sigma — more vol means more chance of a big
favourable move, and the downside is already capped. Monotonicity means a unique
root. And the derivative is vega, which is always positive, so Newton is the
natural method and converges quadratically. Mine hits machine precision in three
steps.

**Q: When does Newton fail?**

When vega collapses — deep in or out of the money. Vega is the denominator of the
step, so a tiny vega means an enormous step and the iterate leaves the valid
range. That's not hypothetical: on a 2.5-week chain, vega spans 130 orders of
magnitude and plain Newton failed at 46 of 70 strikes.

The production answer is a hybrid: Manaster-Koller start (the guess that
*maximises* vega, so you begin where it's most stable), Newton with bracket
guards, and Brent as a fallback because it's bracketing and cannot diverge.

**Q: What's the subtle bug you hit there?**

I converged on `|model price - market price| < 1e-8`. That's wrong. A deep OTM
call worth 2e-9 with vega 2e-7 — any vol within about 1% reproduces that price to
1e-8. So the solver declared success while the volatility was still wrong in its
second decimal place. It gave errors of 1.1e-2 in implied vol, on precisely the
wing options where a surface fit is most fragile.

The fix is that the Newton step `diff/vega` *is* the remaining error in
volatility, to first order. So stop when the step is small, not when the residual
is. Same iteration, correct stopping rule, costs nothing. Worst round-trip error
went from 1.1e-2 to 4.5e-15.

**Q: What if there's no solution?**

Return NaN. Arbitrage bounds require the call price between `max(S·e^(-qT) -
K·e^(-rT), 0)` and `S·e^(-qT)`. Outside that, no volatility reproduces the price.
Real data violates this constantly — stale quotes, wide spreads, mismatched
timestamps — and silently returning garbage instead of NaN is how bad surfaces
get built.

---

## The volatility surface

**Q: Why is the smile there?**

Two non-exclusive reasons. Crash risk: returns have a fat left tail, the
lognormal doesn't, so the market pays up for OTM puts and that excess shows up as
higher implied vol when you invert through a model that can't represent fat
tails. And the leverage effect: when equity falls, leverage and volatility rise
together, so down moves and high vol arrive at the same time. The equity skew was
mild before October 1987 and steep ever since.

**Q: Why SVI rather than a spline?**

A spline through noisy points gives you a surface that permits arbitrage — prices
where someone can construct a guaranteed profit — and then every price you
interpolate off it is wrong. SVI is flexible enough for real smiles, has five
parameters, and its arbitrage conditions are checkable in closed form.

**Q: Why total variance rather than volatility?**

Two reasons. Total variance is the quantity that must be monotone in time for no
calendar arbitrage. And the SVI wings are asymptotically linear in log-moneyness
in total-variance space, which is exactly what Roger Lee's moment formula says
any arbitrage-free smile must do. The parameterisation respects that
automatically.

**Q: What are the arbitrage conditions?**

Butterfly, within one expiry: buy K-dK, sell two K, buy K+dK. The payoff is never
negative so it can't cost less than nothing — equivalently the implied density
must be non-negative. Durrleman's condition states it directly on the smile as
`g(k) >= 0`. If g dips below zero your surface implies a negative probability.

Calendar, across expiries: total implied variance must be non-decreasing in
maturity at fixed log-moneyness. Variance can't un-accumulate, or a calendar
spread is free money.

I check both and report them. I also test the *checker* by feeding it a
deliberately arbitrageable smile — a test that never fails is not a test.

**Q: How do you fit SVI reliably? It's non-convex.**

The key observation: with `y1 = k - m` and `y2 = sqrt((k-m)^2 + sigma^2)`, the
model is `w = a + (b·rho)·y1 + b·y2`. For fixed `(m, sigma)` that's an ordinary
linear regression in three parameters, solvable exactly. So the genuinely
non-convex part is only two-dimensional. I sweep a grid of `(m, sigma)`, solve
the inner linear problem at each node, and seed the full fit from the best few.
That's the idea behind Zeliade's quasi-explicit calibration.

I also floor sigma. As sigma goes to zero the smile degenerates into a sharp V,
which fits noisy points beautifully and is meaningless — it implies an atom of
probability at one strike. It's an attractive local optimum, and without the
floor one of my slices collapsed into it with a 7.6 vol point error while
everything else fitted to 0.1.

**Q: How do you know your surface is right?**

I generate the chain from a *known* SVI surface, with realistic spreads, tick
rounding, thin wings and a few stale quotes. Then I score the fit against the
truth: 0.39 vol points average RMS, and 0.09 on the liquid three-month slice. For
context a typical bid-ask on a liquid option is worth well over half a vol point,
so that's at the noise floor of the data.

**Q: Your shortest expiry has 1.5 vol points of error. Why?**

Not a bug. A one-week option has almost no time value in the wings, so most of
its strikes get filtered out as sub-tick, leaving 10 usable quotes to pin down
five parameters. Real desks have the same problem and handle it the same way:
fewer parameters, or borrow shape from the neighbouring expiry.

**Q: Talk about data cleaning.**

It decides the quality of everything after it. Crossed quotes, zero bids,
sub-$0.05 prices where one tick of quote movement is worth several vol points,
relative spreads over 60% where the "price" is a fiction, no open interest,
extreme moneyness, under three days to expiry, and OTM-only.

The one I'd highlight is "mid must lie between bid and ask". It sounds too
obvious to write. It caught a stale ATM quote implying 71% vol against a true 31%
— invisible to every other filter because the price on its own looked plausible.
That single quote threw an entire slice off by 7 vol points, because vega
weighting gives ATM quotes the *most* influence.

Then robust refitting on top: least squares has a breakdown point of zero, so I
fit, measure residuals, drop anything beyond 4 robust standard deviations scaled
by the MAD — not the standard deviation, because that's itself inflated by the
outliers you're hunting — and refit.

**Q: Why OTM only?**

For ITM options most of the price is intrinsic, so the IV comes out of a small
residual and is noisy. It also sidesteps early exercise for American options. And
you lose nothing: by put-call parity the OTM put and the ITM call carry the same
information, so you keep the liquid one.

**Q: How did you get r and q from real quotes?**

Put-call parity: `C - P = D(F - K)`, model-free, true in any model. That's
linear in K, so in principle a regression gives you both D (hence r) and F
at once. In practice, on SPY's weekly expiries, that regression gave me r
estimates swinging from -23% to +21% between adjacent expiries while F
stayed put the whole time. At a few days to expiry the discount factor is
so close to 1 that the rate signal is smaller than ordinary bid-ask noise —
r just isn't identifiable from short-dated quotes. So I fix r at a known
external rate and solve only for F, which the chain actually can tell you;
q then falls out as `r - ln(F/S)/T`.

**Q: Isn't assuming r a step backward?**

No, because r barely matters at the maturities where it's hardest to pin
down. The price is `e^{-rT}[F·N(d1) - K·N(d2)]` — the *shape* comes from F
alone; r only scales the whole thing by a discount factor that's within a
few basis points of 1 for any sane r at short T. The same fact that makes r
safe to assume is what makes it impossible to extract — you can't measure
what the price barely responds to.

**Q: What did SPY, AAPL, and TSLA actually look like?**

Level and skew moved in opposite directions across the three. ATM vol rose
SPY (11%) → AAPL (24%) → TSLA (39%), which is just diversification: an
index cancels idiosyncratic risk across 500 names, a single stock doesn't.
Skew fell in the *same* order — SPY's short-dated skew was +17.7 vol
points, AAPL's +8.1, TSLA's +1.0, almost flat. That's the more interesting
result: TSLA has by far the most uncertainty and by far the least skew,
because its uncertainty is two-sided idiosyncratic catalyst risk (an
earnings print can send it either way), while SPY's skew is compensation
for one specific, correlated, historically one-directional event — a market
crash, where leverage and volatility rise together and drag every portfolio
down at once. More uncertainty isn't the same question as more *asymmetric*
uncertainty, and the two names answer it oppositely.

**Q: I thought your rho told you the skew direction — did that hold up on
real data?**

Not always, and that's worth knowing precisely because it's a real fitting
pathology, not a hand-wave. One SPY slice fit rho = +0.99 — nominally
call-side skew — while its raw quotes were an obvious, textbook put skew by
eye. The observed strikes for that expiry only spanned a narrow, one-sided
window, so the fitted vertex landed outside any strike that was ever
quoted; inside the window the curve still sloped the right way, but rho's
sign became whatever the optimizer needed, unconstrained by anything the
data actually said. I now track each slice's observed k-range and flag
when the fitted vertex falls outside it, and I read skew off a fixed
moneyness difference (IV at k=-0.10 minus k=+0.10) instead of off rho,
because that number stays meaningful even on a flagged slice.

---

## Monte Carlo

**Q: Why Monte Carlo a European call when there's a formula?**

Two reasons. Validation — you know the answer, so you can check convergence, the
rate, and that your variance reduction is correct rather than plausible. That
earns you the right to trust the engine on something with no closed form. And
most real payoffs have none: Asians, barriers, lookbacks, autocallables.

**Q: What's the convergence rate and can you beat it?**

`s/sqrt(N)`. Ten times the accuracy costs a hundred times the paths. You cannot
beat the rate — you can only shrink `s`. I verify the rate: log-log slope of
standard error against N comes out between -0.4966 and -0.4994 against a theory
of -0.5.

**Q: Explain control variates.**

Alongside the payoff, compute something correlated whose exact mean you know —
here `S_T`, with `E[e^(-rT)S_T] = S·e^(-qT)`. On each path you can see the error
the simulation made on the control, and subtract the corresponding part of the
payoff's error. Unbiased, because the correction has mean zero by construction.
Optimal coefficient `c* = -Cov/Var`, giving `Var × (1 - rho^2)`.

The reason I trust my implementation is that the observed reduction matches
`1/(1 - rho^2)` exactly — rho 0.8779, theory 4.36×, observed 4.36×. That's a much
stronger test than "the variance got smaller".

**Q: Why don't antithetic variates help much for a call?**

Because `max(S_T - K, 0)` is convex in Z, and convexity partly undoes the
cancellation between Z and -Z. Antithetics shine on near-linear payoffs and can
even hurt for strongly convex ones.

**Q: Price an Asian option.**

Arithmetic average — no closed form, because a sum of lognormals isn't lognormal.
But a sum of *normals* is normal, so the geometric-average version is exactly
lognormal and does have a formula. They're correlated above 0.999, so the
geometric Asian is a near-perfect control variate for the arithmetic one: same
paths, one answer known exactly. I get 671× variance reduction.

Three checks that must hold: Asian below vanilla, because averaging cuts
effective volatility; arithmetic at or above geometric, by AM-GM, path by path;
and the controlled and plain estimates agreeing within error bars.

**Q: How do you get Greeks out of Monte Carlo?**

Three ways. Bumping with *different* random numbers is the wrong answer — the
noise swamps the difference. Bumping with common random numbers is much better.
Best is pathwise: differentiate inside the expectation. Since `S_T = S·exp(...)`,
`dS_T/dS = S_T/S`, so `delta = e^(-rT)·E[1{S_T>K}·S_T/S]`. One simulation, no
bump, no differencing noise. Mine lands 0.79 standard errors from analytic.

The catch is it needs a payoff differentiable almost everywhere. Fine for calls
and puts, *not* valid for a digital, whose payoff is a step function — there you
need likelihood ratio or a smoothed payoff.

---

## GARCH

**Q: Why do you need GARCH?**

Because returns are nearly unpredictable but squared returns are strongly
predictable. In my data, autocorrelation of returns is around ±0.02 at every lag,
while autocorrelation of squared returns is 0.24 and decays slowly. Today's move
tells you nothing about tomorrow's direction and a great deal about tomorrow's
magnitude.

**Q: Explain the model.**

`sigma^2_t = omega + alpha·eps^2_(t-1) + beta·sigma^2_(t-1)`. A weighted average
of three things: omega pulls toward the long-run level, alpha times yesterday's
squared surprise is how reactive you are, beta times yesterday's forecast is the
memory that makes shocks decay slowly.

**Q: What's the single most important number in a GARCH fit?**

Persistence, `alpha + beta`. It governs how long a shock lasts — the effect
decays like persistence^h, so the half-life is `ln(0.5)/ln(alpha+beta)`. My fit
is 0.984, a 44-day half-life. That's typical for daily equity data, and it's
exactly why constant-vol Black-Scholes misprices options: today's volatility is
highly informative about next month's.

**Q: How do you handle the constraints?**

Parameter transforms rather than constrained optimisation. `omega = exp(u0)`,
`alpha = 0.999·sigmoid(u1)`, `beta = 0.999·(1-alpha)·sigmoid(u2)`. Every
constraint is satisfied by construction, so the optimiser can never propose a
negative variance or a non-stationary model and never has to crawl along a
boundary. Much more robust than penalties.

I also scale the returns to percent before fitting. Daily returns are ~0.01, so
omega is ~1e-6, and optimisers quietly do badly on numbers that small. That one
line is the difference between reliable convergence and mysterious failures.

**Q: How do you know your MLE is correct?**

Simulate from known parameters and check it recovers them. All four z-scores come
back inside ±0.6. If an estimator can't recover parameters from data generated by
its own model, nothing it says about real data means anything.

I also report standard errors from the numerical Hessian, pushed through to the
constrained parameters with the delta method. A point estimate with no
uncertainty next to it isn't a statistical result.

**Q: Your fit takes 0.2 seconds. What was it before?**

7.28 seconds, which made the rolling out-of-sample study impossible. The
recursion `sigma^2_t = [omega + shock] + beta·sigma^2_(t-1)` is a first-order IIR
filter — the bracket depends only on eps, which is known up front — so it's
exactly `scipy.signal.lfilter([1], [1, -beta], x)`. 36× speedup, bit-identical
output, which I assert in the tests.

**Q: What's the leverage effect?**

Volatility rises more after a fall than after a rally of the same size. Plain
GARCH can't see it because it only uses eps-squared. GJR adds
`gamma·1{eps<0}·eps^2`. My estimate is gamma = 0.081 with t = 5.64, and GJR beats
GARCH on both AIC and BIC.

I also check the negative case: fitted to symmetric simulated data, GJR returns
gamma = 0.0096 with |t| = 0.73. It doesn't invent an effect that isn't there.

**Q: Connect that to the options side.**

They're the same phenomenon. The leverage effect is why out-of-the-money puts are
expensive — down moves come with more volatility, so crash risk is real and gets
priced. GJR's gamma measures it from the returns series; the SVI rho measures it
from the option chain. Two completely different datasets and methodologies
agreeing is the most satisfying result in the project.

**Q: How do you evaluate a volatility forecast?**

Genuinely out of sample: expanding window, refit every 25 days, only data
available at each point. Fitting once on the whole sample and "forecasting" the
same data is in-sample fitting — it always looks great and means nothing.

Against two baselines. EWMA(0.94), which is GARCH with omega = 0 and persistence
exactly 1 — so no long-run level and a flat multi-step forecast, good at one day
and poor at one month. And constant trailing variance, the honest null.

Scored two ways. RMSE, and QLIKE. The true volatility is unobservable, so you
score against a proxy, and the squared return is unbiased but extremely noisy —
`Var(r^2) = 2·sigma^4` for a normal. Patton showed only certain losses still rank
forecasts correctly with a noisy proxy; MSE is one, QLIKE is the other, and QLIKE
is preferred because it's less dominated by a handful of huge days and it
penalises under-forecasting more than over-forecasting, which is the right
asymmetry for risk.

GARCH beats constant by 8.8% RMSE and 11.3% QLIKE over 2,500 out-of-sample days.

**Q: That improvement seems small.**

It is, and that's the proxy, not the model. Most of that RMSE is irreducible
measurement error in the squared-return proxy — no forecast could ever remove it.
Compare models against each other, never against zero. QLIKE, being less
sensitive to the noise, shows the gap more clearly.

---

## The connecting question

**Q: Why is implied volatility higher than your GARCH forecast?**

Because they live under different measures. GARCH estimates volatility under the
real-world measure P — what will actually happen. Implied vol lives under the
risk-neutral measure Q — what the market charges. The difference is the variance
risk premium, and it's one of the most robust findings in empirical finance:
implied exceeds subsequent realised most of the time, across markets and decades.

Mine is about one vol point on a one-month option.

Why would anyone pay it? Because volatility spikes exactly when everything else
in your portfolio is falling. An asset that pays off in a crash is worth more
than its expected payoff, so investors accept a negative expected return to hold
it, and option sellers are paid for supplying it. It's compensation for a nasty
risk profile: small steady gains punctuated by large losses. That's also the
basis of the short-variance trade and why it periodically destroys people.

---

## Things not to say

- "I used Black-Scholes" full stop. Say what you validated and how.
- Quoting a fit RMSE without saying what a bid-ask spread is worth in vol points.
- Claiming your GARCH forecast "beats the market". It doesn't — the gap is a risk
  premium, not an inefficiency.
- Presenting the synthetic chain as real data. Say which, precisely, and say why
  synthetic is *more* informative for validation.
- Claiming r and q are both "extracted from the market" on real data. q is;
  r is a documented assumption, because short-dated quotes don't carry
  enough signal to identify it — say so, and say why that's fine (see "How
  did you get r and q from real quotes?" above).
- Reading SVI's rho as "the skew" without checking whether the fitted vertex
  falls inside the strikes you actually observed. It can have the wrong sign
  and still be the correct answer to the fitting problem it was given.
- "The tests pass" when the tests only assert that functions return numbers. Say
  which identities and recovery properties you test.
