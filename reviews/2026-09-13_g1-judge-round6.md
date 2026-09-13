{
  "ratings": [
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Net share issuance anomaly",
      "rating": 3,
      "one_line_reason": "Textbook Pontiff-Woodgate anomaly; any equity quant lists it immediately."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Liquidity-adjusted momentum",
      "rating": 3,
      "one_line_reason": "Momentum is the most obvious predictor; liquidity adjustment is a standard refinement."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Betting against beta",
      "rating": 3,
      "one_line_reason": "Canonical Frazzini-Pedersen low-beta factor; obvious candidate."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Option-implied skew trading",
      "rating": 4,
      "one_line_reason": "Informed-trading skew signal is less obvious for equities yet documented predictor."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Lead-lag network inference",
      "rating": 4,
      "one_line_reason": "Cross-stock spillover networks are non-obvious but credible for monthly horizon."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Romer-Romer narrative monetary shocks",
      "rating": 3,
      "one_line_reason": "Textbook monetary shock series; first method any macro empiricist names."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Front-door criterion",
      "rating": 2,
      "one_line_reason": "Requires fully mediating mechanism with no confounding; rarely credible in macro-finance."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Shift-share instrumental variables",
      "rating": 2,
      "one_line_reason": "Needs differential exposure shares to aggregate shock; contrived for rate hikes."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Hamilton regime switching model",
      "rating": 1,
      "one_line_reason": "Descriptive switching model; does not identify causal effect of hikes."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Callaway-Sant'Anna group-time DiD",
      "rating": 1,
      "one_line_reason": "No staggered untreated controls under a common rate shock; design invalid."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Hidden Markov model regimes",
      "rating": 3,
      "one_line_reason": "First method anyone names for regimes; obvious and largely backward-looking."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Lead-lag network inference",
      "rating": 4,
      "one_line_reason": "Cross-market spillover leaders can foreshadow vol; non-obvious yet credible."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Truncated realized variance",
      "rating": 2,
      "one_line_reason": "Robust jump-tolerant vol estimator, not an early-warning regime detector."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Trend scanning labeler",
      "rating": 2,
      "one_line_reason": "Trend-labeling tool; weak fit for anticipating volatility regimes."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "GJR-GARCH threshold volatility",
      "rating": 2,
      "one_line_reason": "Reactive backward-looking vol filter; poor for before-the-fact shift detection."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Factor model risk decomposition",
      "rating": 3,
      "one_line_reason": "Standard covariance regularization via factors; immediately obvious."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Independent component analysis",
      "rating": 2,
      "one_line_reason": "Independence assumption fragile for returns; exotic substitute for PCA/factors."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Low volatility anomaly",
      "rating": 1,
      "one_line_reason": "Alpha factor, not a covariance-error robustification technique."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "LASSO variable selection",
      "rating": 1,
      "one_line_reason": "Plain LASSO selects regressors; does not robustify covariance unlike graphical LASSO."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "PageRank centrality ranking",
      "rating": 2,
      "one_line_reason": "Network-centrality diversification is creative but weak vs shrinkage/HRP."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Kalman filter state estimation",
      "rating": 3,
      "one_line_reason": "Canonical state-space denoiser; every quant lists it first."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Continuous wavelet transform scalogram",
      "rating": 4,
      "one_line_reason": "Scale separation for noise is credible and less obvious than MA/Kalman."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Ridge regression",
      "rating": 1,
      "one_line_reason": "Generic regression shrinkage; not a microstructure denoising method."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Hidden Markov model regimes",
      "rating": 1,
      "one_line_reason": "Regime classifier, not a noise filter for intraday signals."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Hodrick-Prescott trend filter",
      "rating": 1,
      "one_line_reason": "Macro quarterly filter with endpoint bias; wrong tool for intraday noise."
    }
  ],
  "summary": {
    "per_problem_avg": [
      3.4,
      1.8,
      2.6,
      1.8,
      2.0
    ],
    "count_ge4_per_problem": [
      2,
      0,
      1,
      0,
      1
    ],
    "count_ge3_per_problem": [
      5,
      1,
      2,
      1,
      2
    ]
  }
}