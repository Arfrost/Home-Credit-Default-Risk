import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from src.preprocessing import get_tree_data
from src.config import RANDOM_STATE

os.makedirs('reports/shap', exist_ok=True)

# ── Load model and data ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, spw = get_tree_data()
xgb_model = joblib.load('models/xgboost.joblib')

print("Computing SHAP values...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values shape: {shap_values.shape}")

# ── 1. Global Feature Importance ───────────────────────────────────────────────
print("\nPlotting global feature importance...")
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values, X_test,
    plot_type='bar',
    max_display=20,
    show=False
)
plt.title('SHAP Feature Importance — Top 20 Features', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('reports/shap/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/shap/shap_importance.png")

# ── 2. SHAP Beeswarm Plot ──────────────────────────────────────────────────────
print("\nPlotting beeswarm...")
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values, X_test,
    max_display=20,
    show=False
)
plt.title('SHAP Beeswarm — Feature Impact Distribution',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('reports/shap/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/shap/shap_beeswarm.png")

# ── 3. Local Explanation — High Risk Applicant ────────────────────────────────
print("\nPlotting local explanations...")

# Find a high risk defaulter (actual default, high predicted probability)
y_prob = xgb_model.predict_proba(X_test)[:, 1]
default_mask = y_test.values == 1
high_risk_idx = np.where(default_mask)[0][np.argmax(y_prob[default_mask])]

plt.figure(figsize=(14, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[high_risk_idx],
        feature_names=X_test.columns.tolist()
    ),
    max_display=15,
    show=False
)
plt.title('SHAP Waterfall — High Risk Applicant (Actual Default)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/shap/shap_waterfall_highrisk.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/shap/shap_waterfall_highrisk.png")

# Find a low risk applicant (actual non-default, low predicted probability)
nondefault_mask = y_test.values == 0
low_risk_idx = np.where(nondefault_mask)[0][np.argmin(y_prob[nondefault_mask])]

plt.figure(figsize=(14, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[low_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[low_risk_idx],
        feature_names=X_test.columns.tolist()
    ),
    max_display=15,
    show=False
)
plt.title('SHAP Waterfall — Low Risk Applicant (Actual Non-Default)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/shap/shap_waterfall_lowrisk.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/shap/shap_waterfall_lowrisk.png")

# ── 4. Top Features Summary ────────────────────────────────────────────────────
mean_shap = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 15 Features by Mean |SHAP|:")
print(mean_shap.head(15).to_string(index=False))

mean_shap.to_csv('reports/shap/shap_feature_importance.csv', index=False)
print("\nSaved: reports/shap/shap_feature_importance.csv")
print("\nSHAP analysis complete.")