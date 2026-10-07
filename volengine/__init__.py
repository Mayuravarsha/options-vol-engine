"""volengine -- options pricing and volatility modelling.

Modules
-------
black_scholes  Black-Scholes-Merton pricing, Greeks, put-call parity, and
               finite-difference Greeks used to validate the closed forms.
implied_vol    Newton-Raphson inversion with a Manaster-Koller start and a
               Brent bracketing fallback, plus no-arbitrage bound checks.
monte_carlo    GBM simulation with antithetic and control variates, a
               path-dependent Asian pricer, pathwise Greeks, and a
               convergence study.
surface        SVI volatility surface fitting with butterfly (Durrleman) and
               calendar arbitrage checks.
garch          GARCH(1,1) and GJR-GARCH by maximum likelihood, forecasting,
               and an out-of-sample comparison against EWMA and constant vol.
data           A synthetic option chain with known ground truth, cleaning
               filters, and optional yfinance loaders for real data.
plots          Figures.
"""

from .black_scholes import (bs_greeks, bs_price, finite_difference_greeks,  # noqa: F401
                            forward, put_call_parity_gap)
from .implied_vol import (implied_vol, implied_vol_brent,  # noqa: F401
                          implied_vol_newton, implied_vol_vector,
                          intrinsic_bounds)
from .monte_carlo import (convergence_study, geometric_asian_call,  # noqa: F401
                          mc_asian_call, mc_european, mc_pathwise_delta)
from .surface import (SVIParams, VolSurface, build_surface,  # noqa: F401
                      check_butterfly, check_calendar, durrleman_g,
                      fit_svi_slice)
from .garch import (GarchFit, annualised_vol_forecast, ewma_variance,  # noqa: F401
                    fit_garch, forecast_variance, news_impact_curve, qlike,
                    rolling_forecast_study, simulate_garch)
from .data import (DEFAULT_TRUE_SVI, apply_forward_rate, clean_chain,  # noqa: F401
                   generate_synthetic_chain, implied_forward_rate,
                   load_cached_chain, load_returns_csv, load_yfinance_chain,
                   load_yfinance_returns, true_iv)

__version__ = "0.1.0"
