import joblib
import numpy as np
import pandas as pd
import json
import torch
from src.preprocessing import get_tree_data, load_data, add_features, NEW_FEATURES, CreditRiskANN
from src.config import MODEL_DIR

# Load data and models
X_train, X_test, y_train, y_test, spw = get_tree_data()
xgb_model  = joblib.load(f'{MODEL_DIR}xgboost.joblib')
lr_bundle  = joblib.load(f'{MODEL_DIR}logistic_regression.joblib')
ann_bundle = joblib.load(f'{MODEL_DIR}ann.joblib')

with open('src/feature_sets.json') as f:
    feature_sets = json.load(f)

# Get XGBoost probabilities on test set
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Find 3 representative observations
# High risk — actual default, high predicted probability
default_idx   = np.where(y_test.values == 1)[0]
high_risk_idx = default_idx[np.argsort(y_prob_xgb[default_idx])[-1]]

# Medium risk — closest to 0.5 probability
medium_risk_idx = np.argmin(np.abs(y_prob_xgb - 0.5))

# Low risk — actual non-default, low predicted probability
nondefault_idx = np.where(y_test.values == 0)[0]
low_risk_idx   = nondefault_idx[np.argmin(y_prob_xgb[nondefault_idx])]

indices = {
    'High Risk':   high_risk_idx,
    'Medium Risk': medium_risk_idx,
    'Low Risk':    low_risk_idx
}

# LR predictions
lr_new  = [f for f in NEW_FEATURES if f not in ['EXT_SOURCE_1x2','EXT_SOURCE_2x3','EXT_SOURCE_1x3']]
X_lr    = X_test.reindex(columns=feature_sets['lr_only'] + lr_new, fill_value=0)
X_lr_s  = lr_bundle['scaler'].transform(X_lr)
y_prob_lr = lr_bundle['model'].predict_proba(X_lr_s)[:, 1]

# ANN predictions
X_ann_s = ann_bundle['scaler'].transform(X_test)
ann_bundle['model'].eval()
with torch.no_grad():
    y_prob_ann = ann_bundle['model'](
        torch.FloatTensor(X_ann_s)
    ).squeeze().numpy()

# Credit scores
FACTOR   = 20 / np.log(2)
log_odds = lr_bundle['model'].intercept_[0] + X_lr_s.dot(lr_bundle['model'].coef_[0])
scores   = (-FACTOR * log_odds + 600).astype(int)

# Print results
print(f"{'='*65}")
print(f"{'Observation':15s} {'Actual':8s} {'XGBoost':10s} {'LR':10s} {'ANN':10s} {'Score':8s}")
print(f"{'='*65}")

for label, idx in indices.items():
    actual   = int(y_test.values[idx])
    p_xgb    = y_prob_xgb[idx]
    p_lr     = y_prob_lr[idx]
    p_ann    = y_prob_ann[idx]
    score    = scores[idx]
    print(f"{label:15s} {actual:8d} {p_xgb:10.4f} {p_lr:10.4f} {p_ann:10.4f} {score:8d}")

print(f"{'='*65}")
print("\nKey features for each observation:")

top_features = ['EXT_SOURCES_MEAN', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'LOAN_PAYMENT_LENGTH', 'inst_late_payment_rate',
                'YEARS_EMPLOYED', 'AGE_YEARS', 'bur_active_ratio',
                'prev_refusal_rate', 'cc_avg_utilization_6m']

for label, idx in indices.items():
    print(f"\n{label}:")
    for feat in top_features:
        if feat in X_test.columns:
            print(f"  {feat:35s}: {X_test.iloc[idx][feat]:.4f}")