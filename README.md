# ETF Mean-Reversion & Regime Prediction

A systematic signal research project investigating short-horizon mean reversion, time-series structure, and market regimes in liquid US ETFs using Python.

The project follows an end-to-end quantitative research workflow:

**data cleaning → feature engineering → statistical testing → predictive modelling → signal construction → out-of-sample backtesting**

The main objective is not simply to maximise predictive accuracy, but to understand whether statistically measurable relationships in ETF returns are stable, economically meaningful, and robust out of sample.

---

## Research Questions

This project focuses on several related questions:

1. Do daily ETF returns exhibit short-term autocorrelation or mean reversion?
2. Are extreme price deviations followed by statistically significant reversals?
3. How do return distributions differ from the Gaussian assumption?
4. Which technical and statistical features contain information about next-day returns?
5. Can regression and classification models improve on simple statistical signals?
6. Does signal performance depend on the prevailing volatility regime?
7. Do relationships remain profitable after transaction costs and out-of-sample testing?

---

## Assets

The initial analysis uses liquid US equity ETFs:

* SPY — S&P 500
* QQQ — Nasdaq-100
* IWM — Russell 2000

Additional ETFs can later be introduced for cross-asset and PCA analysis, including:

* XLF — Financials
* XLK — Technology
* XLE — Energy

Daily OHLCV data are used over approximately 5–10 years.

---

## Feature Engineering

The initial feature set includes:

* 1-day return
* 5-day return
* 20-day return
* 20-day rolling volatility
* 20-day moving-average distance
* 20-day price z-score
* 20-day volume z-score
* lagged daily returns

For example, the rolling price z-score is defined as

$$
z_t =
\frac{P_t-\mu_{20,t}}
{\sigma_{20,t}}
$$

where the mean and standard deviation use only information available up to time \(t\).

This avoids look-ahead bias.

The main regression target is

$$
y_t = r_{t+1}
$$

and the classification target is

$$
y_t =
\mathbf{1}(r_{t+1}>0).
$$

---

## 1. Statistical Analysis — `statsmodels`

The first stage studies the time-series properties of ETF returns.

### OLS Regression

Example model:

$$
r_{t+1}
=
\beta_0
+
\beta_1 r_t
+
\beta_2 r_{t-1}
+
\epsilon_{t+1}
$$

The analysis focuses on:

* regression coefficients
* t-statistics
* p-values
* confidence intervals
* \(R^2\)
* residual diagnostics

Additional regressions test whether variables such as price z-score, recent momentum, volatility, and volume contain information about future returns.

### Stationarity

Augmented Dickey–Fuller tests are applied to:

* price levels
* daily returns
* moving-average deviations

The purpose is to compare non-stationary price levels with more stationary transformations such as returns and relative price deviations.

### Autocorrelation

ACF and PACF are used to investigate serial dependence in returns.

The project also compares the autocorrelation of

$$
r_t
$$

with the autocorrelation of

$$
r_t^2.
$$

This helps identify volatility clustering even when raw return autocorrelation is weak.

---

## 2. Statistical Testing — `SciPy`

`scipy.stats` is used to investigate return distributions and signal significance.

### Distribution Analysis

Daily ETF returns are evaluated using:

* skewness
* excess kurtosis
* Jarque–Bera test
* normality tests
* fitted probability distributions

The analysis examines the extent to which empirical ETF returns exhibit fat tails and departures from Gaussian assumptions.

### Mean-Reversion Tests

A simple mean-reversion hypothesis is:

$$
z_t < -2
\quad \Rightarrow \quad
E[r_{t+1}] > 0
$$

and

$$
z_t > 2
\quad \Rightarrow \quad
E[r_{t+1}] < 0.
$$

One-sample t-tests are used to determine whether conditional future returns are statistically different from zero.

### Information Coefficient

Spearman rank correlation is used to measure the monotonic relationship between features and future returns.

Examples include:

* 1-day return vs next-day return
* 5-day momentum vs next-day return
* volatility vs next-day return
* z-score vs next-day return

This provides a simple time-series analogue of a rank-based information coefficient.

---

## 3. Machine Learning — `scikit-learn`

The prediction stage compares traditional statistical models with a standard machine-learning workflow.

### Regression

Models include:

* Linear Regression
* Ridge Regression
* Lasso Regression

The target is:

$$
r_{t+1}
$$

Candidate predictors include:

* recent returns
* rolling volatility
* moving-average distance
* price z-score
* volume z-score
* lagged returns

This section highlights the distinction between:

**`statsmodels` for statistical inference**

and

**`scikit-learn` for predictive modelling and model selection.**

---

## 4. Classification

The target is converted into a directional variable:

$$
y_t =
\mathbf{1}(r_{t+1}>0)
$$

Models include:

* Logistic Regression
* Random Forest Classifier

Evaluation metrics include:

* accuracy
* precision
* ROC-AUC
* confusion matrix

Because directional accuracy alone does not determine trading profitability, model outputs are later converted into positions and evaluated through portfolio-level metrics.

---

## 5. Time-Series Cross-Validation

Random shuffling is avoided.

Instead, the project uses chronological train, validation, and test periods together with:

```python
from sklearn.model_selection import TimeSeriesSplit
```

The objective is to prevent future information from leaking into model training.

A typical structure is:

```text
Train       → earlier historical period
Validation  → subsequent period
Test        → final untouched period
```

All preprocessing and model fitting are performed using information available at the relevant point in time.

---

## 6. Market Regime Analysis

Market conditions are not assumed to be constant through time.

A simple volatility regime can be defined using rolling volatility:

```text
Low-volatility regime
Normal regime
High-volatility regime
```

The analysis investigates whether mean-reversion signals behave differently across regimes.

A later extension treats regime prediction as a classification problem using variables such as:

* recent volatility
* recent returns
* volume
* cross-ETF behaviour
* momentum measures

This allows trading signals to depend on both the strength of a mean-reversion signal and the prevailing market environment.

---

## 7. PCA Across ETFs

Principal Component Analysis is applied to returns from multiple ETFs.

```python
from sklearn.decomposition import PCA
```

The analysis investigates whether a small number of latent factors explain a substantial fraction of cross-ETF return variation.

The first principal component is interpreted as a candidate broad market mode where supported by the empirical loadings.

---

## 8. Backtesting

Model predictions and statistical signals are converted into trading positions.

A basic backtest follows:

```python
position = signal.shift(1)
strategy_return = position * market_return
```

The lag ensures that a signal computed at time \(t\) is not applied to a return that has already occurred.

Performance metrics include:

* cumulative return
* annualised return
* annualised volatility
* Sharpe ratio
* maximum drawdown
* turnover

Transaction costs are incorporated using:

```python
transaction_cost = turnover * cost_per_trade
```

The final comparison will evaluate:

| Model / Signal         | Sharpe | Annualised Return | Max Drawdown | Turnover |
| ---------------------- | -----: | ----------------: | -----------: | -------: |
| Z-score Mean Reversion |    TBD |               TBD |          TBD |      TBD |
| OLS                    |    TBD |               TBD |          TBD |      TBD |
| Ridge                  |    TBD |               TBD |          TBD |      TBD |
| Logistic Regression    |    TBD |               TBD |          TBD |      TBD |
| Random Forest          |    TBD |               TBD |          TBD |      TBD |

Results will be populated only after the out-of-sample analysis is completed.

---

## Repository Structure

```text
systematic-signal-research/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_features.ipynb
│   ├── 02_statsmodels_analysis.ipynb
│   ├── 03_scipy_statistics.ipynb
│   ├── 04_ml_regression.ipynb
│   ├── 05_ml_classification.ipynb
│   └── 06_pca_backtest.ipynb
│
├── src/
│   ├── features.py
│   ├── backtest.py
│   └── metrics.py
│
├── requirements.txt
└── README.md
```

---

## Python Stack

The project uses:

```text
Python
NumPy
pandas
SciPy
statsmodels
scikit-learn
Matplotlib
yfinance
```

---

## Research Principles

The project follows several basic principles of quantitative research:

* Do not use future information when constructing features.
* Do not randomly shuffle financial time-series observations.
* Separate statistical significance from economic significance.
* Compare models against simple baselines.
* Evaluate signals out of sample.
* Include transaction costs.
* Avoid tuning strategies repeatedly on the final test set.
* Treat unusually strong predictive results as a reason to check for data leakage.

---

## Project Status

Work in progress.

Completed:

* Data collection and cleaning
* Return and rolling-feature construction
* Lagged-feature construction
* Initial mean-reversion signal definition
* OLS and time-series diagnostics

In progress:

* Statistical hypothesis testing
* Machine-learning models
* Time-series cross-validation
* Regime analysis
* PCA
* Out-of-sample backtesting

---

## Final Deliverable

The completed project will summarise the research as:

**Research question → Data → Features → Statistical tests → Models → Out-of-sample results → Limitations**

The goal is to evaluate whether simple ETF signals exhibit reproducible predictive structure rather than to optimise an in-sample trading strategy.
