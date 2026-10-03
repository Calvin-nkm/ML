# ML
Predicting NVDA next-day direction with Random Forest, GBM, XGBoost &amp; LightGBM. 52 technical features narrowed to 18 via a Filter → Wrapper → Embedded funnel, tuned with time-series CV and evaluated on a chronological hold-out. Includes entropy/information gain walkthrough. Python, scikit-learn, MLflow.
# NVDA Direction Prediction: Gradient Boosting & Ensemble Methods

A machine learning pipeline that predicts whether **NVIDIA (NVDA)** will have a meaningful positive move on the next trading day, using technical indicators, a three-stage feature-selection funnel, and four tuned tree-based classifiers.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Models](https://img.shields.io/badge/Models-RF%20%7C%20GBM%20%7C%20XGBoost%20%7C%20LightGBM-green)

---

## Overview

| Item | Detail |
|---|---|
| **Ticker** | NVDA |
| **Data** | Daily OHLCV via `yfinance` (auto-adjusted), May 2019 – Apr 2026 |
| **Task** | Binary classification: next-day UP (1) vs DOWN (0) |
| **Label** | Next-day log return > +0.3% → 1, < −0.3% → 0; days within ±0.3% are discarded as noise |
| **Samples** | 1,380 labelled days (757 UP / 623 DOWN, ~54.9% positive) |
| **Split** | Chronological 80/20. Train: Apr 2020 – Jan 2025 (1,104 rows). Test: Feb 2025 – Apr 2026 (276 rows) |
| **Validation** | `TimeSeriesSplit` (5 folds), scored on AUC-ROC |
| **Tracking** | MLflow |

The notebook is structured in three sections:

1. **Section A: Entropy in Classification.** Shannon entropy and information gain, demonstrated numerically on a real RSI14 split of NVDA data.
2. **Section B: Feature Selection (Funnelling).** 52 engineered features narrowed to 18 via Filter → Wrapper → Embedded stages.
3. **Section C: Model Building, Tuning & Evaluation.** Four models tuned with `RandomizedSearchCV` and compared on a held-out test set.

---

## Feature Engineering

52 features across 15 technical groups, all computed from OHLCV data:

| Group | Examples |
|---|---|
| Intraday range | Open–Close, High–Low (absolute and % of close) |
| Return direction | Sign of 1/3/5-day returns |
| Lagged returns | Lag 1, 2, 3, 5, 10 |
| Momentum | 3/5/10/20-day momentum (absolute and %) |
| Moving averages | SMA 5/10/20/50, EMA 5/10/20, and crossover ratios |
| Volatility | Rolling std of returns (5/10/20-day), volatility ratio |
| Oscillators | RSI (7/14/21), MACD (line, signal, histogram), Stochastic %K/%D, Williams %R |
| Bands & range | Bollinger %B, ATR14, ATR % |
| Volume | Volume change, volume/SMA ratios, price-volume trend |
| Price position | Price vs SMA20/50, position within 52-week range |

---

## Feature Selection: The Funnelling Approach

```
52 features
   │  Stage 1a  Variance Threshold          → 52
   │  Stage 1b  Correlation Filter (>0.92)  → 37
   │  Stage 1c  Mutual Information          → 37
   │  Stage 2   RFE (Gradient Boosting)     → 18
   │  Stage 3   Embedded GBM importances    → 18
   ▼
18 final features (confirmed by two-round ablation study)
```

**Design choices**

- **Methods deliberately excluded:** Chi-Square (needs non-negative inputs), ANOVA (assumes normality), Lasso/Ridge (assumes linearity), and Boruta (shuffling conflicts with temporal ordering).
- **Protected features:** `RSI7`, `RSI14`, `RSI21`, `MACD`, `MACD_hist`, `ATR_pct` are retained at every stage on domain-knowledge grounds, since RSI (overbought/oversold) and MACD (trend momentum) capture complementary information.
- **Ablation study:** Two rounds (±2 features, then ±1) with time-series CV confirmed 18 as the optimal feature count.

**Final feature set (18):** `Lag2`, `Lag3`, `Lag5`, `Lag10`, `PV_trend`, `SMA5`, `Vol_chg`, `Vol20`, `ATR_pct`, `SMA_ratio_10_50`, `SMA_ratio_5_20`, `Vol_SMA_ratio`, `MACD`, `MACD_hist`, `RSI7`, `RSI14`, `RSI21`, `Mom20`

---

## Models

All models are tuned with `RandomizedSearchCV` (40 iterations, 5-fold `TimeSeriesSplit`, AUC-ROC scoring).

| Model | Role |
|---|---|
| Random Forest | Bagging baseline (`class_weight='balanced'`) |
| Sklearn Gradient Boosting | Classic boosting |
| XGBoost | Regularised boosting (L1/L2, column subsampling) |
| LightGBM | Leaf-wise histogram-based boosting |

---

## Results

Held-out test set (276 days, Feb 2025 – Apr 2026):

| Model | CV AUC | Test AUC | Precision (UP) | Recall (UP) | F1 (UP) | Accuracy |
|---|---|---|---|---|---|---|
| Random Forest | 0.5219 | **0.5592** | 0.635 | 0.429 | 0.512 | 0.544 |
| Sklearn GBM | 0.5196 | 0.5252 | 0.592 | 0.377 | 0.460 | 0.507 |
| XGBoost | 0.5194 | 0.5473 | 0.589 | 0.688 | 0.635 | 0.558 |
| LightGBM | 0.5111 | 0.5590 | 0.602 | 0.442 | 0.509 | 0.525 |

**Takeaway:** Predictive power is modest. AUCs sit only slightly above the 0.50 random baseline, which is typical for daily equity direction prediction where signal-to-noise is very low. Random Forest and LightGBM score highest on test AUC; XGBoost achieves the best F1 and accuracy by predicting UP more often.

---

## Limitations

- No macro or cross-asset features (e.g. VIX, SOX semiconductor index, rate expectations).
- Single-stock, single-regime test window. The 2023–24 AI-driven rally is underrepresented in training.
- AUCs are close to 0.50, so results should not be interpreted as a tradable edge.
- The simplified backtest excludes transaction costs, slippage, and market impact.

---

## Getting Started

```bash
git clone https://github.com/Calvin-nkm/<repo-name>.git
cd <repo-name>
pip install yfinance xgboost lightgbm scikit-learn mlflow pandas numpy matplotlib seaborn
jupyter notebook NVDA_ML.ipynb
```

Run the cells in order. The first code cell downloads the data and builds all features, and every later section depends on it. Because data is pulled live from Yahoo Finance, re-running with a later end date will change the numbers.

---

## Repository Structure

```
├── NVDA_ML.ipynb    # Full pipeline: entropy, feature selection, modelling, evaluation
└── README.md
```

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · MLflow · yfinance · Matplotlib · Seaborn

---

## Disclaimer

This project is for educational and research purposes only. It is not investment advice.

## Author

**Calvin**: Market Risk professional, Singapore · [GitHub](https://github.com/Calvin-nkm)
