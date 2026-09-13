{
  "ratings": [
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Idiosyncratic volatility anomaly",
      "rating": 3,
      "one_line_reason": "Textbook Ang et al low-IVOL effect; any equity quant lists it first."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Standardized unexpected earnings",
      "rating": 3,
      "one_line_reason": "Canonical PEAD/SUE factor; obvious earnings-momentum candidate."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Book-to-market value",
      "rating": 3,
      "one_line_reason": "Fama-French value staple; immediately obvious."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Variational autoencoder alpha",
      "rating": 4,
      "one_line_reason": "Latent-factor autoencoder in Gu-Kelly-Xiu spirit; niche twist but plausible."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Pastor-Stambaugh liquidity beta",
      "rating": 4,
      "one_line_reason": "Known liquidity premium; less immediate than value/mom but fits monthly horizon."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Momentum with crowding adjustment",
      "rating": 4,
      "one_line_reason": "Base momentum obvious; crowding filter is the non-obvious plausibly useful add."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Ridge regression",
      "rating": 3,
      "one_line_reason": "Standard penalized cross-sectional workhorse; obvious baseline."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Huber robust regression",
      "rating": 4,
      "one_line_reason": "Outlier-robust fitting for noisy cross-section; sensible niche pick."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Net share issuance anomaly",
      "rating": 3,
      "one_line_reason": "Well-known Pontiff-Woodgate anomaly; standard zoo entry."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Cross-sectional z-score stat arb",
      "rating": 2,
      "one_line_reason": "Daily stat-arb framing mismatched to monthly horizon; at best reversal proxy."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Information discreteness momentum",
      "rating": 4,
      "one_line_reason": "Frog-in-pan ID effect; genuinely niche yet documented for monthly returns."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Option-implied skew trading",
      "rating": 4,
      "one_line_reason": "Xing-Zhang style informed-skew predictor; non-obvious cross-asset signal for spot."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Cross-sectional momentum",
      "rating": 3,
      "one_line_reason": "Jegadeesh-Titman momentum; most obvious monthly predictor."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Overnight returns anomaly",
      "rating": 4,
      "one_line_reason": "Lou-Polk overnight-intraday split; clever non-consensus monthly signal."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Betting against beta",
      "rating": 3,
      "one_line_reason": "Frazzini-Pedersen BAB; standard low-risk factor."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Elastic net regularization",
      "rating": 3,
      "one_line_reason": "Obvious high-dimensional regularizer for factor zoo."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Earnings yield value",
      "rating": 3,
      "one_line_reason": "Standard value variant; immediately obvious."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Gross profitability factor",
      "rating": 3,
      "one_line_reason": "Novy-Marx Fama-French profitability; obvious quality leg."
    },
    {
      "problem": "predict cross-sectional equity returns for the next month",
      "method": "Lead-lag network inference",
      "rating": 2,
      "one_line_reason": "Lead-lag mostly decays within days; stretch for one-month cross-section."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Vector error correction model",
      "rating": 2,
      "one_line_reason": "Captures cointegration dynamics but does not identify causal hike effect."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Generalized method of moments",
      "rating": 2,
      "one_line_reason": "Generic estimator, not an identification strategy; needs valid instruments."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "PCMCI time-series causal discovery",
      "rating": 2,
      "one_line_reason": "Needs long stationary series; macro hikes too few and confounded for discovery."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Local average treatment effect",
      "rating": 1,
      "one_line_reason": "An estimand not a method; listing it without design is a category error."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Causal forests",
      "rating": 1,
      "one_line_reason": "Needs unit-level treatment variation; aggregate hike hits all sectors."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Two-stage least squares instrumental variables",
      "rating": 3,
      "one_line_reason": "Textbook IV route with Fed-surprise instrument; obvious but valid baseline."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Front-door criterion",
      "rating": 1,
      "one_line_reason": "No credible fully-mediating variable isolates rates to returns; infeasible here."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Triple differences",
      "rating": 2,
      "one_line_reason": "No never-treated sector block; needs cross-country third leg to even apply."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Romer-Romer narrative monetary shocks",
      "rating": 5,
      "one_line_reason": "Narrative-identified exogenous shocks plus local projections; exactly the credible design."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Vector autoregression",
      "rating": 3,
      "one_line_reason": "Structural VAR for monetary transmission; standard obvious macro baseline."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Doubly robust estimation",
      "rating": 2,
      "one_line_reason": "Cross-sectional DR needs propensity and overlap; time-series shock violates setup."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Double machine learning",
      "rating": 2,
      "one_line_reason": "High-dim partialling cannot fix simultaneity and anticipation with few hikes."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Quantile regression",
      "rating": 1,
      "one_line_reason": "Describes conditional quantiles, solves no endogeneity; wrong tool for causality."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Shift-share instrumental variables",
      "rating": 2,
      "one_line_reason": "Bartik logic fits regions, not sector equity response; forced exposure construction."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "OLS with Newey-West HAC standard errors",
      "rating": 1,
      "one_line_reason": "Correlation with robust SEs; simultaneity and forward-looking prices kill causality."
    },
    {
      "problem": "estimate the causal effect of interest rate hikes on sector equity returns",
      "method": "Panel fixed effects estimator",
      "rating": 2,
      "one_line_reason": "Time FE absorbs the hike, no time FE leaves macro confounds; cannot identify shock."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Variance risk premium",
      "rating": 4,
      "one_line_reason": "IV-minus-RV fear gauge often shifts pre-turbulence; sensible non-consensus lead."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Smooth transition autoregression",
      "rating": 4,
      "one_line_reason": "Smooth-transition AR captures gradual parameter change; niche but legitimate lens."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Markov-switching multifractal model",
      "rating": 4,
      "one_line_reason": "Calvet-Fisher multifractal fits long-memory vol cascades; exotic yet relevant."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "L1 trend filtering",
      "rating": 2,
      "one_line_reason": "Piecewise-linear break smoother; backward-looking, not early warning."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Empirical mode decomposition",
      "rating": 2,
      "one_line_reason": "Adaptive ex-post decomposition with mode mixing; no predictive regime mechanism."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "VPIN order-flow toxicity",
      "rating": 5,
      "one_line_reason": "Bulk-volume toxicity flagged fragility before flash crash; ideal microstructure lead."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Spectral entropy measure",
      "rating": 2,
      "one_line_reason": "Generic complexity heuristic; thin evidence linking entropy to equity vol regimes."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Option-implied skew trading",
      "rating": 3,
      "one_line_reason": "Skew steepening reflects crash pricing; well-known fear gauge for stress."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Delta-hedged option returns",
      "rating": 2,
      "one_line_reason": "Isolates variance premium ex-post; indirect noisy proxy, not a detector."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Hidden Markov model regimes",
      "rating": 3,
      "one_line_reason": "Canonical ex-post regime classifier; obvious baseline every vol team lists."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Factor timing regime switching",
      "rating": 2,
      "one_line_reason": "Times factor premia, not market vol; wrong target for vol early warning."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Particle filter state estimation",
      "rating": 4,
      "one_line_reason": "Sequential nonlinear inference for SV-jump models; strong real-time tracker."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Continuous wavelet transform scalogram",
      "rating": 2,
      "one_line_reason": "Time-frequency view localizes bursts contemporaneously; does not forecast shifts."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Quantile regression",
      "rating": 1,
      "one_line_reason": "Static conditional-quantile fit; no dynamics or break detection."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Realized variance and bipower variation",
      "rating": 2,
      "one_line_reason": "Measures current jump vs diffusive vol; measurement input, not forward signal."
    },
    {
      "problem": "detect regime shifts in market volatility before they fully materialize",
      "method": "Hamilton regime switching model",
      "rating": 3,
      "one_line_reason": "Textbook Markov-switching AR; obvious ex-post regime workhorse."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Black-Litterman allocation",
      "rating": 2,
      "one_line_reason": "Shrinks expected returns to equilibrium; does not clean covariance error."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Elastic net regularization",
      "rating": 2,
      "one_line_reason": "Regression penalty needs portfolio reformulation; indirect weight shrinkage."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Hayashi-Yoshida asynchronous covariance",
      "rating": 1,
      "one_line_reason": "Fixes nonsynchronous tick covariance; irrelevant to high-dim robustness."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Hamilton regime switching model",
      "rating": 1,
      "one_line_reason": "Time-series regime model; does not address covariance sampling error."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Discrete wavelet transform denoising",
      "rating": 1,
      "one_line_reason": "Time-series signal smoother; not a covariance estimator or allocation rule."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Marchenko-Pastur eigenvalue clipping",
      "rating": 4,
      "one_line_reason": "RMT bulk-edge split directly cleans sampling noise; purpose-built fix."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Independent component analysis",
      "rating": 1,
      "one_line_reason": "Seeks independent sources, unstable on returns; no error-robustness property."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Markowitz mean-variance optimization",
      "rating": 1,
      "one_line_reason": "Unconstrained mean-variance is the error-maximizer; the disease not the cure."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Factor model risk decomposition",
      "rating": 3,
      "one_line_reason": "BARRA-style low-rank covariance cuts dimension; standard obvious robustifier."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Ridge regression",
      "rating": 2,
      "one_line_reason": "Resembles shrinkage only after reformulation; generic label as listed."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Graph Laplacian regularization",
      "rating": 2,
      "one_line_reason": "Smoothness over ad-hoc asset graph; researchy with thin portfolio evidence."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Minimum spanning tree asset clustering",
      "rating": 4,
      "one_line_reason": "Mantegna clustering underlies HRP hierarchy; avoids inverting noisy covariance."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Low volatility anomaly",
      "rating": 1,
      "one_line_reason": "Return factor, not an estimator; says nothing about covariance robustness."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Realized variance and bipower variation",
      "rating": 1,
      "one_line_reason": "Univariate jump-robust vol; not a high-dim covariance solution."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "Variational autoencoder alpha",
      "rating": 1,
      "one_line_reason": "Return-prediction generative model; irrelevant to allocation robustness."
    },
    {
      "problem": "build a diversified allocation robust to covariance estimation error",
      "method": "GARCH volatility model",
      "rating": 1,
      "one_line_reason": "Univariate vol dynamics; does not fix cross-sectional covariance error."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Variational autoencoder alpha",
      "rating": 2,
      "one_line_reason": "Generic latent net with no microstructure structure; overkill for tick filtering."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Pre-averaged realized volatility",
      "rating": 2,
      "one_line_reason": "Estimates integrated vol under noise; vol estimator not signal cleaner."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Huber robust regression",
      "rating": 2,
      "one_line_reason": "Downweights bad ticks but ignores bounce autocorrelation; partial fix."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "VPIN order-flow toxicity",
      "rating": 1,
      "one_line_reason": "Toxicity gauge, not a filter; measures flow, does not denoise."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Factor model risk decomposition",
      "rating": 1,
      "one_line_reason": "Strips systematic risk, not bid-ask bounce; wrong noise concept."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Discrete wavelet transform denoising",
      "rating": 4,
      "one_line_reason": "Thresholded wavelet shrinkage is purpose-built for bursty tick noise."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Unscented Kalman filter",
      "rating": 4,
      "one_line_reason": "Nonlinear state-space tracker for true price plus noise; natural filter."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Truncated realized variance",
      "rating": 1,
      "one_line_reason": "Jump-truncated vol estimator; outputs vol number, not cleaned signal."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Double machine learning",
      "rating": 1,
      "one_line_reason": "Causal ML for treatment effects; category error for denoising."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Kalman filter state estimation",
      "rating": 3,
      "one_line_reason": "Linear true-price plus noise workhorse; obvious correct baseline."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Short-term reversal",
      "rating": 1,
      "one_line_reason": "Exploits bounce as alpha; confuses trading noise with removing it."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Empirical mode decomposition",
      "rating": 2,
      "one_line_reason": "Adaptive mode sifting without theory; fragile ad-hoc tick denoiser."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Independent component analysis",
      "rating": 2,
      "one_line_reason": "Needs multichannel independent sources; forced fit for single signal."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Ridge regression",
      "rating": 1,
      "one_line_reason": "Global L2 shrinkage; no tick-dynamics model for bounce or staleness."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Roll implicit spread estimator",
      "rating": 2,
      "one_line_reason": "Estimates spread from autocovariance; diagnoses noise, does not filter series."
    },
    {
      "problem": "remove microstructure noise from an intraday alpha signal",
      "method": "Continuous wavelet transform scalogram",
      "rating": 1,
      "one_line_reason": "Time-frequency picture; visualization without denoising rule."
    }
  ],
  "summary": {
    "per_problem_avg": [
      3.263,
      2.0,
      2.813,
      1.75,
      1.875
    ],
    "count_ge3_per_problem": [
      17,
      3,
      8,
      3,
      3
    ],
    "distinct_methods_rated_per_problem": [
      19,
      16,
      16,
      16,
      16
    ]
  }
}