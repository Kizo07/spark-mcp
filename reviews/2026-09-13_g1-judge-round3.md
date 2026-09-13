{
  "ratings": [
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Net share issuance anomaly",
      "rating": 3,
      "one_line_reason": "Standard anomaly-zoo factor any quant lists; plausible but obvious and more annual than monthly."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Post-earnings-announcement drift",
      "rating": 3,
      "one_line_reason": "Textbook monthly cross-sectional signal; strong fit but immediately obvious."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Dividend yield premium",
      "rating": 3,
      "one_line_reason": "Classic value proxy; plausible cross-sectionally but obvious and weak at one-month horizon."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Overnight returns anomaly",
      "rating": 4,
      "one_line_reason": "Second-tier signal using overnight-vs-intraday decomposition; not first-listed but plausible monthly predictor."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Low volatility anomaly",
      "rating": 3,
      "one_line_reason": "Canonical cross-sectional factor; fits but among the first five any team names."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Negative controls and placebo testing",
      "rating": 2,
      "one_line_reason": "Useful validation falsification, not an estimator of the causal effect itself."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Granger causality test",
      "rating": 2,
      "one_line_reason": "Predictive precedence, not structural causation; wrong tool with forward-looking asset prices."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Empirical mode decomposition",
      "rating": 1,
      "one_line_reason": "Signal decomposition with no identification strategy for policy causation."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Black-Litterman allocation",
      "rating": 1,
      "one_line_reason": "Portfolio construction method; has no role in estimating rate-onto-returns causation."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Discrete wavelet transform denoising",
      "rating": 1,
      "one_line_reason": "Denoising filter, not a causal estimator; off-target for hike effects."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Delta-hedged option returns",
      "rating": 4,
      "one_line_reason": "Option-implied variance information can front-run realized vol; creative and genuinely relevant."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Amihud illiquidity premium",
      "rating": 2,
      "one_line_reason": "Liquidity dries into stress but equity illiquidity factor is a stretched early-warning proxy."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Short-term reversal",
      "rating": 1,
      "one_line_reason": "Cross-sectional equity signal with no leading vol-regime mechanism."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Net share issuance anomaly",
      "rating": 1,
      "one_line_reason": "Slow corporate-finance anomaly; nonsense as a vol-regime early detector."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Bollinger band mean reversion",
      "rating": 2,
      "one_line_reason": "Volatility-adaptive bands but lagging retail indicator, not anticipatory regime detection."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Black-Litterman allocation",
      "rating": 3,
      "one_line_reason": "Obvious allocation stabilizer but targets mean uncertainty, not covariance error like shrinkage would."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Generalized method of moments",
      "rating": 1,
      "one_line_reason": "Generic estimation framework, not a robust-allocation construction."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Vector error correction model",
      "rating": 1,
      "one_line_reason": "Cointegration time-series model; irrelevant to covariance-robust weighting."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "PC algorithm causal discovery",
      "rating": 2,
      "one_line_reason": "Causal graph for diversification is creative but PC is fragile and not covariance-robust."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Momentum with crowding adjustment",
      "rating": 1,
      "one_line_reason": "Single alpha signal, not an allocation robust to estimation error."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Independent component analysis",
      "rating": 2,
      "one_line_reason": "Blind source separation mismatches bid-ask bounce structure; Kalman/wavelets dominate."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Amihud illiquidity premium",
      "rating": 1,
      "one_line_reason": "Cross-sectional liquidity factor, not a denoising filter."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Cross-sectional momentum",
      "rating": 1,
      "one_line_reason": "An alpha signal itself, not a microstructure-noise remover."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Accruals anomaly",
      "rating": 1,
      "one_line_reason": "Low-frequency accounting anomaly; nonsense for intraday microstructure."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Net share issuance anomaly",
      "rating": 1,
      "one_line_reason": "Annual corporate-action factor; entirely off-target for tick-level denoising."
    }
  ],
  "summary": {
    "per_problem_avg": [
      3.2,
      1.4,
      2.0,
      1.6,
      1.2
    ],
    "count_ge4_per_problem": [
      1,
      0,
      1,
      0,
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