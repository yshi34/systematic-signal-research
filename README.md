# ETF Mean-Reversion & Regime Prediction

A systematic quantitative research project investigating short-horizon mean reversion, time-series dependence, return distributions, and market regimes in liquid US ETFs using Python.

The project follows an end-to-end research workflow:

**data cleaning → feature engineering → statistical analysis → predictive modelling → signal construction → out-of-sample backtesting**

The objective is not simply to maximise predictive accuracy, but to determine whether apparent patterns in ETF returns are statistically supported, economically meaningful, and robust out of sample.

---

## Research Questions

The project investigates several related questions:

1. Do daily ETF returns exhibit short-term autocorrelation or mean reversion?
2. Are extreme deviations from recent price levels followed by statistically significant reversals?
3. Are daily ETF returns approximately Gaussian, or do they exhibit skewness and fat tails?
4. Which features contain measurable information about next-day returns?
5. Can regression and classification models improve on simple statistical signals?
6. Does signal performance depend on the prevailing volatility regime?
7. Do predictive relationships remain useful after transaction costs and out-of-sample testing?

---

## Assets

The initial analysis focuses on liquid US equity ETFs:

- **SPY** — S&P 500
- **QQQ** — Nasdaq-100
- **IWM** — Russell 2000

The first stage of the project focuses primarily on SPY before extending the analysis across multiple ETFs.

Additional ETFs may later be introduced for PCA and cross-asset analysis:

- XLF — Financials
- XLK — Technology
- XLE — Energy

Daily OHLCV data are used over approximately 5–10 years.

---

# 1. Data & Feature Engineering

The raw daily ETF data are transformed into a set of predictive and descriptive features.

The initial feature set includes:

- 1-day return
- 5-day return
- 20-day return
- 20-day rolling volatility
- 20-day moving-average distance
- 20-day price z-score
- 20-day volume z-score
- lagged daily returns

Daily returns are defined as:

\[
r_t =
\frac{P_t}{P_{t-1}} - 1
\]

Multi-period returns are defined similarly:

\[
r_t^{(k)}
=
\frac{P_t}{P_{t-k}} - 1
\]

Rolling annualised volatility is calculated from daily return volatility:

\[
\sigma_{\text{annual}}
=
\sigma_{\text{daily}}\sqrt{252}
\]

---

## Rolling Price Z-Score

A central feature in the mean-reversion analysis is the rolling price z-score:

\[
z_t =
\frac{P_t-\mu_{20,t}}
{\sigma_{20,t}}
\]

where both the rolling mean and rolling standard deviation use only information available up to time \(t\).

This avoids look-ahead bias.

Interpretation:

```text
large negative z-score → unusually low relative price
large positive z-score → unusually high relative price
```

A simple mean-reversion hypothesis is therefore:

\[
z_t < -2
\Rightarrow
E[r_{t+1}] > 0
\]

and

\[
z_t > 2
\Rightarrow
E[r_{t+1}] < 0
\]

---

## Prediction Targets

The regression target is:

\[
y_t = r_{t+1}
\]

The classification target is:

\[
y_t =
\mathbf{1}(r_{t+1}>0)
\]

Each observation therefore follows the structure:

\[
X_t \rightarrow r_{t+1}
\]

where all features in \(X_t\) are known by time \(t\).

---

# 2. Time-Series Analysis — `statsmodels`

The second stage investigates whether historical ETF returns contain measurable serial structure.

---

## OLS Regression

A baseline autoregressive-style model is:

\[
r_{t+1}
=
\beta_0
+
\beta_1r_t
+
\beta_2r_{t-1}
+
\epsilon_{t+1}
\]

The analysis focuses on:

- regression coefficients
- t-statistics
- p-values
- confidence intervals
- \(R^2\)
- residual behaviour

Additional multivariate regressions introduce features such as:

- recent momentum
- rolling volatility
- price z-score
- volume z-score

The purpose is to distinguish between:

```text
coefficient sign
```

which describes the estimated direction of a relationship,

and

```text
statistical significance
```

which measures how strongly the data support that relationship.

---

## Stationarity — Augmented Dickey–Fuller Test

The Augmented Dickey–Fuller test is applied to:

- ETF price levels
- daily returns
- moving-average deviations

The null hypothesis is:

\[
H_0:
\text{the series contains a unit root}
\]

so a sufficiently small p-value provides evidence against non-stationarity.

The analysis illustrates the common distinction between:

```text
price levels → often non-stationary
returns → typically more stationary
```

---

## Autocorrelation

The autocorrelation function is used to estimate:

\[
\rho_k =
Corr(r_t,r_{t-k})
\]

for multiple lags.

This helps determine whether returns exhibit:

- short-term continuation
- short-term reversal
- little serial dependence

Partial autocorrelation is also examined to separate direct lag relationships from correlations transmitted through intermediate lags.

---

## Return vs Squared-Return Autocorrelation

The project compares the autocorrelation of:

\[
r_t
\]

with:

\[
r_t^2
\]

Raw returns may exhibit weak autocorrelation even when squared returns show persistent dependence.

This provides an empirical way to study:

\[
\textbf{volatility clustering}
\]

where large price movements tend to be followed by other large movements, even if their direction is difficult to predict.

---

## Autoregressive Models

Simple autoregressive models are estimated using:

```python
from statsmodels.tsa.ar_model import AutoReg
```

For example:

\[
r_t =
c + \phi_1r_{t-1}+\epsilon_t
\]

The analysis connects autoregression to ordinary regression on lagged values and uses ACF/PACF diagnostics to examine possible lag structure.

---

# 3. Statistical Testing — `SciPy`

The third stage investigates whether observed return patterns are statistically distinguishable from random variation.

---

## Return Distribution

Daily SPY returns are analysed using:

- mean
- standard deviation
- skewness
- excess kurtosis
- empirical return histograms

The objective is to evaluate how closely the empirical return distribution resembles a Gaussian distribution.

---

## Normality Tests

Two statistical tests are used:

### Jarque–Bera Test

The Jarque–Bera test examines departures from normality using:

- skewness
- kurtosis

The null hypothesis is:

\[
H_0:
\text{returns are consistent with normality}
\]

---

### D'Agostino Normality Test

The project also uses:

```python
scipy.stats.normaltest
```

as an additional test of whether daily returns are consistent with a Gaussian distribution.

These tests are combined with empirical skewness and excess kurtosis to investigate:

\[
\textbf{fat tails}
\]

in financial returns.

---

## Mean-Reversion Significance Test

Extreme rolling z-scores are used to define simple conditional samples.

For example:

\[
z_t < -2
\]

defines an oversold sample.

The corresponding future returns are:

\[
r_{t+1}
\mid
z_t<-2
\]

A one-sample t-test examines:

\[
H_0:
E[r_{t+1}\mid z_t<-2]=0
\]

against a mean-reversion alternative:

\[
H_1:
E[r_{t+1}\mid z_t<-2]>0
\]

Similarly, overbought observations can be tested using:

\[
z_t>2
\]

with the alternative:

\[
E[r_{t+1}\mid z_t>2]<0
\]

This distinguishes between an observed positive or negative conditional return and evidence that the effect is statistically different from zero.

---

## Statistical vs Economic Significance

A central principle of the project is that:

\[
\boxed{
\text{statistical significance}
\neq
\text{economic significance}
}
\]

A small p-value does not necessarily imply that a trading strategy is profitable.

A statistically detectable signal may still be too small to survive:

- transaction costs
- bid-ask spreads
- turnover
- market impact
- model instability

Economic significance will therefore be evaluated separately during the backtesting stage.

---

## Spearman Rank Correlation

The relationship between predictive features and next-day returns is also evaluated using Spearman correlation:

\[
\rho_s =
Corr(
Rank(X_t),
Rank(r_{t+1})
)
\]

Candidate features include:

- 1-day return
- 5-day return
- 20-day return
- rolling volatility
- price z-score
- volume z-score

Spearman correlation is useful because it measures monotonic rather than strictly linear relationships.

In a broader quantitative research context, rank correlation between a signal and future returns is closely related to the concept of an:

\[
\textbf{Information Coefficient (IC)}
\]

---

## Conditional Z-Score Analysis

Z-score observations are grouped into buckets such as:

```text
z < -2
-2 ≤ z < -1
-1 ≤ z < 0
0 ≤ z < 1
1 ≤ z < 2
z > 2
```

For each group, the analysis compares:

- average next-day return
- median next-day return
- volatility
- sample size

This provides a visual and statistical way to examine whether future returns vary systematically with the degree of price deviation.

---

# 4. Machine Learning Regression

The next stage will use `scikit-learn` to construct predictive models.

Models will include:

- Linear Regression
- Ridge Regression
- Lasso Regression

Candidate predictors include:

```text
ret_1d
ret_5d
ret_20d
vol_20d
ma_20_dist
zscore_20d
volume_zscore_20d
lagged returns
```

The target remains:

\[
r_{t+1}
\]

This stage will highlight the distinction between:

**`statsmodels` → statistical inference**

and

**`scikit-learn` → prediction, regularisation, model selection and pipelines**

---

# 5. Classification

The next-day target will also be transformed into:

\[
y_t=
\mathbf{1}(r_{t+1}>0)
\]

Models will include:

- Logistic Regression
- Random Forest Classifier

Evaluation metrics will include:

- accuracy
- precision
- ROC-AUC
- confusion matrix

Model performance will be compared against simple baselines rather than against 50% accuracy alone.

---

# 6. Time-Series Cross-Validation

Random train/test shuffling will not be used for the financial time series.

Instead, the project will use chronological splits together with:

```python
from sklearn.model_selection import TimeSeriesSplit
```

A typical structure is:

```text
Train → Validation → Test
```

where the test period remains untouched during model development.

This is designed to prevent:

\[
\boxed{\text{look-ahead bias}}
\]

and other forms of information leakage.

---

# 7. Market Regime Prediction

Market conditions are not assumed to remain constant through time.

A basic regime definition will initially focus on rolling volatility:

```text
Low-volatility regime
Normal regime
High-volatility regime
```

The project will investigate whether mean-reversion signals behave differently across these environments.

A classification model may then estimate:

\[
P(
\text{high-volatility regime}_{t+1}
\mid X_t
)
\]

using information such as:

- recent volatility
- momentum
- lagged returns
- volume
- cross-ETF behaviour

The objective is to investigate whether trading signals should depend both on:

```text
signal strength
```

and:

```text
market regime
```

---

# 8. PCA Across ETFs

Principal Component Analysis will be applied to returns across several ETFs using:

```python
from sklearn.decomposition import PCA
```

The analysis will study whether a small number of latent factors explain a substantial fraction of cross-ETF variation.

The first principal component will be examined as a potential broad market factor, subject to the empirical factor loadings.

---

# 9. Backtesting

Statistical signals and machine-learning predictions will ultimately be converted into trading positions.

A basic backtest follows:

```python
position = signal.shift(1)

strategy_return = (
    position * market_return
)
```

The lag ensures that a signal computed using information at time \(t\) is only applied to future returns.

Performance metrics will include:

- cumulative return
- annualised return
- annualised volatility
- Sharpe ratio
- maximum drawdown
- turnover

Transaction costs will also be included:

```python
transaction_cost = (
    turnover * cost_per_trade
)
```

---

## Final Model Comparison

The final project will compare several signals and models:

| Model / Signal | Sharpe | Annualised Return | Max Drawdown | Turnover |
|---|---:|---:|---:|---:|
| Z-Score Mean Reversion | TBD | TBD | TBD | TBD |
| OLS | TBD | TBD | TBD | TBD |
| Ridge | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |

Only out-of-sample results will be used for the final comparison.

---

# Repository Structure

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

# Python Stack

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

# Research Principles

The project follows several principles intended to reduce common quantitative research errors:

- Use only information available at the prediction time when constructing features.
- Avoid random shuffling of financial time-series observations.
- Separate statistical significance from economic significance.
- Compare predictive models against simple baselines.
- Keep an untouched out-of-sample test period.
- Include transaction costs in strategy evaluation.
- Avoid repeated parameter optimisation on the final test set.
- Treat unexpectedly strong predictive performance as a reason to check for leakage.
- Distinguish exploratory analysis from confirmatory statistical testing.
- Avoid selecting thresholds solely because they produce attractive in-sample p-values or Sharpe ratios.

---

# Project Status

### Completed

- [x] ETF data collection and cleaning
- [x] Daily and multi-period return construction
- [x] Rolling volatility features
- [x] Moving-average distance
- [x] Rolling price and volume z-scores
- [x] Lagged-return features
- [x] Regression and classification target construction
- [x] OLS regression with `statsmodels`
- [x] Coefficient, t-statistic, p-value and confidence-interval interpretation
- [x] Residual analysis
- [x] Augmented Dickey–Fuller stationarity testing
- [x] ACF and PACF analysis
- [x] Return and squared-return autocorrelation comparison
- [x] Basic autoregressive modelling
- [x] Return skewness and kurtosis analysis
- [x] Jarque–Bera normality testing
- [x] Additional normality testing with SciPy
- [x] Mean-reversion t-tests
- [x] Spearman rank correlation / signal IC analysis
- [x] Conditional z-score bucket analysis

### In Progress

- [ ] Linear Regression with `scikit-learn`
- [ ] Ridge and Lasso regularisation
- [ ] Feature scaling and ML pipelines
- [ ] Logistic Regression
- [ ] Random Forest classification
- [ ] Time-series cross-validation
- [ ] Market regime prediction
- [ ] PCA across ETFs
- [ ] Out-of-sample backtesting
- [ ] Transaction-cost modelling
- [ ] Final model comparison

---

# Current Research Workflow

The project currently follows:

\[
\boxed{
\text{Data}
\rightarrow
\text{Features}
\rightarrow
\text{Time-Series Diagnostics}
\rightarrow
\text{Statistical Tests}
\rightarrow
\text{Machine Learning}
\rightarrow
\text{Backtest}
}
\]

The first three stages are now complete.

The next stage focuses on building predictive regression models using `scikit-learn` and evaluating whether regularisation improves out-of-sample performance.

---

# Final Deliverable

The completed project will present the research in the following structure:

**Research question → Data → Features → Statistical evidence → Predictive models → Out-of-sample results → Economic performance → Limitations**

The goal is to evaluate whether simple ETF signals exhibit reproducible predictive structure rather than to optimise an in-sample trading strategy.