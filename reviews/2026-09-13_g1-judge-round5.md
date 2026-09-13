{
  "ratings": [
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Net share issuance anomaly",
      "rating": 3,
      "one_line_reason": "Classic issuance factor; every quant lists it for cross-section."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Option-implied skew trading",
      "rating": 4,
      "one_line_reason": "Skew predicts cross-section; option-to-equity link less obvious but fits 1M."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Lead-lag network inference",
      "rating": 4,
      "one_line_reason": "Economic-link spillover at 1M plausible; network framing beyond standard momentum."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Asset growth anomaly",
      "rating": 3,
      "one_line_reason": "Textbook investment factor; obvious factor-zoo entry."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Elastic net regularization",
      "rating": 3,
      "one_line_reason": "Standard high-dim combiner for factor zoo; obvious tooling."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Doubly robust estimation",
      "rating": 3,
      "one_line_reason": "Generic DML/AIPW with macro controls defensible but standard; Fed endogeneity unresolved."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Panel fixed effects estimator",
      "rating": 2,
      "one_line_reason": "Time FE absorbs common hike shock; sector-FE-only version still endogenous."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Synthetic control method",
      "rating": 2,
      "one_line_reason": "No untreated donor when all sectors treated together; needs cross-country stretch."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Fama-MacBeth cross-sectional regression",
      "rating": 1,
      "one_line_reason": "Estimates cross-sectional premia, cannot identify a common time-series shock."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Hidden Markov model regimes",
      "rating": 1,
      "one_line_reason": "Describes regimes, provides no causal identification of hike effect."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Factor timing regime switching",
      "rating": 3,
      "one_line_reason": "Regime-switching to find regimes is circular and the first thing listed."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "GARCH volatility model",
      "rating": 2,
      "one_line_reason": "Backward-looking vol filter; lags breaks, cannot anticipate shifts."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Variational autoencoder alpha",
      "rating": 1,
      "one_line_reason": "Return-prediction autoencoder, not a vol-regime early-warning detector."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "PCMCI time-series causal discovery",
      "rating": 2,
      "one_line_reason": "Finds lagged drivers that could lead vol, but not a regime-shift detector."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Vector autoregression",
      "rating": 2,
      "one_line_reason": "Linear constant-parameter dynamics; needs MS extension to see regimes."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Factor model risk decomposition",
      "rating": 3,
      "one_line_reason": "Barra-style structured covariance is the textbook answer to estimation error."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Kelly criterion sizing",
      "rating": 1,
      "one_line_reason": "Growth-optimal sizing amplifies estimation error; opposite of robust."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Variational autoencoder alpha",
      "rating": 1,
      "one_line_reason": "Alpha signal model has no link to covariance-robust allocation."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Kalman filter state estimation",
      "rating": 1,
      "one_line_reason": "State filter adds specification burden; not a robustness device."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Spectral entropy measure",
      "rating": 1,
      "one_line_reason": "Signal-complexity metric irrelevant to portfolio construction."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Variational autoencoder alpha",
      "rating": 2,
      "one_line_reason": "Generic autoencoder denoising; overkill and unstructured for bid-ask bounce."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Fast Fourier transform spectral analysis",
      "rating": 3,
      "one_line_reason": "Low-pass filtering of high-frequency bounce is standard denoising."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Liquidity-adjusted momentum",
      "rating": 1,
      "one_line_reason": "Alpha factor, not a tick-noise filter."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "LASSO variable selection",
      "rating": 1,
      "one_line_reason": "Selects variables, does not filter microstructure noise."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Lead-lag network inference",
      "rating": 1,
      "one_line_reason": "Cross-asset lead-lag unrelated to single-signal denoising."
    }
  ],
  "summary": {
    "per_problem_avg": [
      3.4,
      1.8,
      2.0,
      1.4,
      1.6
    ],
    "count_ge4_per_problem": [
      2,
      0,
      0,
      0,
      0
    ],
    "count_ge3_per_problem": [
      5,
      1,
      1,
      1,
      1
    ]
  }
}