{
  "ratings": [
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Residual momentum",
      "rating": 3,
      "one_line_reason": "Standard momentum variant any equity team lists first; fits but textbook obvious."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Standardized unexpected earnings",
      "rating": 3,
      "one_line_reason": "Classic PEAD/SUE factor; strong fit but immediately obvious for monthly CS."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Book-to-market value",
      "rating": 3,
      "one_line_reason": "Fama-French staple; plausible but most obvious value proxy."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Gross profitability factor",
      "rating": 3,
      "one_line_reason": "Standard quality/profitability factor; fits but textbook obvious."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Betting against beta",
      "rating": 3,
      "one_line_reason": "Well-known low-beta anomaly; plausible but standard list item."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Front-door criterion",
      "rating": 2,
      "one_line_reason": "Clever causal idea but complete mediation implausible for rates-sectors; huge identification stretch."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Elastic net regularization",
      "rating": 2,
      "one_line_reason": "Prediction regularizer not causal identification; at best nuisance estimator in DML pipeline."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Term structure carry",
      "rating": 1,
      "one_line_reason": "Rates trading strategy not a causal estimator; category error."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Propensity score matching",
      "rating": 1,
      "one_line_reason": "Requires cross-sectional overlap/binary treatment; invalid for universal time-series macro shock."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "EGARCH asymmetric volatility",
      "rating": 1,
      "one_line_reason": "Univariate vol model; does not identify causal mean effect."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Option-implied skew trading",
      "rating": 3,
      "one_line_reason": "Forward-looking skew/term structure is sensible early warning but first thing vol desk lists."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Liquidity-adjusted momentum",
      "rating": 1,
      "one_line_reason": "Equity CS factor; liquidity-vol link too indirect, momentum transform irrelevant."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Dividend yield premium",
      "rating": 1,
      "one_line_reason": "Slow fundamental value signal; cannot lead fast vol regime shifts."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Low volatility anomaly",
      "rating": 1,
      "one_line_reason": "Shares word volatility but is CS stock selection, not regime timing model."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Index reconstitution arbitrage",
      "rating": 1,
      "one_line_reason": "Event arb around index changes; unrelated to vol regime prediction."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Marchenko-Pastur eigenvalue clipping",
      "rating": 4,
      "one_line_reason": "Direct RMT covariance cleaner; strong fit and one step beyond default Ledoit-Wolf."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "EGARCH asymmetric volatility",
      "rating": 1,
      "one_line_reason": "Univariate dynamics add estimation error; does not robustify covariance."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Factor timing regime switching",
      "rating": 1,
      "one_line_reason": "Adds timing/estimation risk; opposite of robustification."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Quantile regression",
      "rating": 1,
      "one_line_reason": "Tail modeling tool, not a covariance-error solution."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Pastor-Stambaugh liquidity beta",
      "rating": 1,
      "one_line_reason": "Liquidity risk premium, unrelated to covariance estimation robustness."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Variational autoencoder alpha",
      "rating": 2,
      "one_line_reason": "Generic nonlinear denoiser; plausible but ignores tick-structure theory and leakage/latency risks."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Residual momentum",
      "rating": 1,
      "one_line_reason": "Monthly equity factor; irrelevant to tick microstructure filtering."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Post-earnings-announcement drift",
      "rating": 1,
      "one_line_reason": "Low-frequency event drift; unrelated to intraday noise removal."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Quality minus junk",
      "rating": 1,
      "one_line_reason": "Low-frequency quality factor; category error for microstructure."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "MAX lottery effect",
      "rating": 1,
      "one_line_reason": "Cross-sectional lottery anomaly; nonsense for noise filtering."
    }
  ],
  "summary": {
    "per_problem_avg": [
      3.0,
      1.4,
      1.4,
      1.6,
      1.2
    ],
    "count_ge4_per_problem": [
      0,
      0,
      0,
      1,
      0
    ],
    "count_ge3_per_problem": [
      5,
      0,
      1,
      1,
      0
    ]
  }
}