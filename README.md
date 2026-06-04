# Home Credit Default Risk — End-to-End ML System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.787-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

An end-to-end credit default risk modelling system built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The project covers SQL-based feature engineering, statistical feature selection, multi-model training with Optuna hyperparameter optimization, SHAP explainability, a credit scorecard, and a live Streamlit deployment.

🚀 **Live App:** [home-credit-default-risk-arfrost.streamlit.app](https://home-credit-default-risk-arfrost.streamlit.app)

---

## Team

| Name | Role |
|---|---|
| [Arfrost] | ML Pipeline, Optuna Tuning, Technical Lead |
| [demirdemirdemirdemir] | SQL Feature Engineering, DuckDB |
| [berkw2b] | Statistical EDA, Feature Analysis |
| [Arfrost] | Model Evaluation, Streamlit App |

Marmara University — Statistics Department, Graduation Project 2026

---

## Strategy

The project is structured as a layered system where each stage feeds into the next.

### Stage 1 — SQL Feature Engineering
All feature engineering is done in SQL using DuckDB before any modelling. Seven relational tables (~56 million rows total) are aggregated to the `SK_ID_CURR` level using a strict **aggregate first, join second** principle.

- POS Cash, Installments, Credit Card, Bureau tables each aggregated separately
- Time-windowed features: 3-month, 6-month windows for POS/CC; 1-year for installments
- Cross-table ratio features computed after final merge (debt-to-income, annuity-to-income etc.)
- Kaggle 17th place solution features added: `EXT_SOURCES_MEAN`, `LOAN_PAYMENT_LENGTH`, `EXT_SOURCE_2x3`
- Final dataset: **307,511 rows × 230 features** → compressed to **61.4 MB** (Parquet + float32)

### Stage 2 — Statistical Feature Selection
A four-step selection pipeline reduces 230 features to model-specific sets:

| Step | Method | Before | After |
|---|---|---|---|
| 1 | Boruta (150k sample, target-encoded categoricals) | 230 | 150 |
| 2 | VIF analysis (two rounds) | 150 | 108 |
| 3 | KS significance test | 108 | 106 |
| 4 | New feature addition | 106 | 114 |

Three separate feature sets are maintained per model sensitivity:

| Set | Models | Features |
|---|---|---|
| `full` | XGBoost, ANN | 114 |
| `reduced` | SVM | 101 |
| `lr_only` | Logistic Regression | 84 |

**Key findings:** EXT_SOURCE_2 had the highest KS statistic (0.2233). Two features (AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK) were removed with KS ≈ 0.0009.

### Stage 3 — Modelling
Four models trained with Optuna TPE optimization. Class imbalance handled per-model:

| Model | Imbalance Strategy | Feature Set | Trials |
|---|---|---|---|
| Logistic Regression | SMOTE (strategy=0.3) | lr_only (84) | 50 |
| SVM | SMOTE + 10k subsample* | reduced (101) | 20 |
| XGBoost | scale_pos_weight=11.39 | full (114) | 50 |
| ANN | Loss weighting (pos_w=11.39) | full (114) | 30 |

*SVM O(n²) complexity — 10k stratified subsample documented as methodological constraint.

ANN architecture: `Input(114) → [Linear → BatchNorm → ReLU → Dropout] × n → Sigmoid(1)`

A **policy rule layer** sits on top of model scores:
- Rule 1: `AMT_CREDIT / AMT_INCOME > 15` → force high risk
- Rule 2: `EXT_SOURCE_1 = EXT_SOURCE_2 = EXT_SOURCE_3 = 0` → force high risk

### Stage 4 — Evaluation & Explainability
- **DeLong test** on all model pairs — all comparisons statistically significant (p < 0.05)
- **SHAP TreeExplainer** on XGBoost — global importance, beeswarm, waterfall plots
- **Credit scorecard** derived from Logistic Regression coefficients (PDO=20, Base=600)

---

## Results

### Model Performance

| Model | AUC-ROC | KS | Gini | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** ★ | **0.7866** | **0.4352** | **0.5732** | 0.6824 | 0.3026 |
| ANN | 0.7712 | 0.4149 | 0.5424 | 0.4149 | 0.3308 |
| Logistic Regression | 0.7609 | 0.3901 | 0.5218 | **0.6860** | 0.2708 |
| SVM | 0.7603 | 0.3968 | 0.5205 | 0.2465 | 0.2726 |

### vs Industry Standards

| Metric | This Project | Industry Standard | Assessment |
|---|---|---|---|
| AUC-ROC | 0.787 | 0.72 – 0.82 |  Production quality |
| KS | 0.435 | > 0.40 excellent |  Excellent |
| Gini | 0.573 | > 0.40 acceptable |  Good |
| Recall | 0.682 | > 0.60 target |  Above target |

### DeLong Test — All Pairs Significant (p < 0.05)

| Comparison | ΔAUC | Significant |
|---|---|---|
| XGBoost vs ANN | 0.0154 | Yes |
| XGBoost vs LR | 0.0257 | Yes |
| XGBoost vs SVM | 0.0263 | Yes |
| ANN vs LR | 0.0103 | Yes |

### Top 10 Features (SHAP)

| Rank | Feature | Mean \|SHAP\| | Source |
|---|---|---|---|
| 1 | EXT_SOURCES_MEAN | 0.392 | Engineered |
| 2 | LOAN_PAYMENT_LENGTH | 0.159 | Engineered |
| 3 | pos_avg_instalment_future | 0.145 | POS table |
| 4 | EXT_SOURCE_2x3 | 0.132 | Engineered |
| 5 | CODE_GENDER | 0.121 | Main table |
| 6 | pos_total_months | 0.116 | POS table |
| 7 | prev_credit_to_app_ratio | 0.094 | Previous app |
| 8 | YEARS_EMPLOYED | 0.091 | Main table |
| 9 | inst_late_payment_rate | 0.086 | Installments |
| 10 | OWN_CAR_AGE | 0.083 | Main table |

3 of the top 4 features are engineered in this project. Adding `EXT_SOURCES_MEAN` alone improved AUC from 0.7833 → 0.7874 (+0.004).

### Scorecard

| Group | Mean Score |
|---|---|
| Non-Default | 616.4 |
| Default | 585.0 |
| Separation | +31.4 points |

---

## Stack
Python 3.10 · DuckDB · Pandas · Scikit-learn · XGBoost
PyTorch · Optuna · SHAP · imbalanced-learn · Streamlit
Apache Parquet · joblib

---

## Setup

```bash
git clone https://github.com/Arfrost/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSVs in `data/raw/`.

```bash
# Run preprocessing
python -m src.preprocessing

# Train tree models (XGBoost + ANN)
python -m src.train_tree

# Train LR + SVM
python -m src.train_lr_svm

# Full evaluation
python -m src.evaluate

# SHAP analysis
python -m src.shap_analysis

# Scorecard
python -m src.scorecard

# Launch app
streamlit run app/streamlit_app.py
```
