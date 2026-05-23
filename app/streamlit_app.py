import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import torch
import shap
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# ── Dynamic Path Configuration ────────────────────────────────────────────────
# Get the absolute path of the directory containing this file (app/)
CURRENT_DIR = Path(__file__).resolve().parent
# Go up one level to the project root directory
PROJECT_ROOT = CURRENT_DIR.parent

# Safely inject the project root into sys.path to resolve internal src imports
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import add_features, NEW_FEATURES, CreditRiskANN

# Define robust absolute paths for your assets
MODELS_DIR_PATH = PROJECT_ROOT / "models"
FEATURE_SETS_JSON_PATH = PROJECT_ROOT / "src" / "feature_sets.json"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide"
)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Use cross-platform path joining to completely prevent directory string mismatches
    xgb_model  = joblib.load(MODELS_DIR_PATH / "xgboost.joblib")
    lr_bundle  = joblib.load(MODELS_DIR_PATH / "logistic_regression.joblib")
    ann_bundle = joblib.load(MODELS_DIR_PATH / "ann.joblib")
    
    with open(FEATURE_SETS_JSON_PATH) as f:
        feature_sets = json.load(f)
        
    return xgb_model, lr_bundle, ann_bundle, feature_sets

xgb_model, lr_bundle, ann_bundle, feature_sets = load_models()

# ── SHAP Explainer ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(xgb_model)

explainer = load_explainer()

# ── Helper ─────────────────────────────────────────────────────────────────────
def predict_all(input_df):
    df = add_features(input_df.copy())
    for col in NEW_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # XGBoost
    X_xgb    = df[feature_sets['full'] + NEW_FEATURES].fillna(0)
    prob_xgb = xgb_model.predict_proba(X_xgb)[0, 1]

    # LR
    lr_new   = [f for f in NEW_FEATURES if f not in
                ['EXT_SOURCE_1x2','EXT_SOURCE_2x3','EXT_SOURCE_1x3']]
    X_lr     = df[feature_sets['lr_only'] + lr_new].fillna(0)
    X_lr_s   = lr_bundle['scaler'].transform(X_lr)
    prob_lr  = lr_bundle['model'].predict_proba(X_lr_s)[0, 1]

    # ANN
    X_ann    = df[feature_sets['full'] + NEW_FEATURES].fillna(0)
    X_ann_s  = ann_bundle['scaler'].transform(X_ann)
    ann_bundle['model'].eval()
    with torch.no_grad():
        prob_ann = ann_bundle['model'](
            torch.FloatTensor(X_ann_s)
        ).squeeze().item()

    # Credit Score from LR
    FACTOR = 20 / np.log(2)
    OFFSET = 600
    log_odds = lr_bundle['model'].intercept_[0] + X_lr_s.dot(lr_bundle['model'].coef_[0])
    score = int(-FACTOR * log_odds[0] + OFFSET)

    return prob_xgb, prob_lr, prob_ann, score, X_xgb

def risk_color(prob):
    if prob < 0.3:   return '🟢', 'Low Risk'
    elif prob < 0.6: return '🟡', 'Medium Risk'
    else:            return '🔴', 'High Risk'

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Risk System")
page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction", "Model Info"])

# ── Page 1: Single Prediction ──────────────────────────────────────────────────
if page == "Single Prediction":
    st.title("Single Applicant Risk Assessment")
    st.markdown("Enter applicant details to assess credit default risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Information")
        AMT_INCOME_TOTAL  = st.number_input("Annual Income ($)", 10000, 10000000, 150000, step=5000)
        AMT_CREDIT        = st.number_input("Credit Amount ($)", 10000, 4000000, 500000, step=10000)
        AMT_ANNUITY       = st.number_input("Monthly Annuity ($)", 1000, 300000, 25000, step=1000)
        EXT_SOURCE_1      = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
        EXT_SOURCE_2      = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
        EXT_SOURCE_3      = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)

    with col2:
        st.subheader("Personal Information")
        AGE_YEARS         = st.slider("Age", 18, 70, 35)
        YEARS_EMPLOYED    = st.slider("Years Employed", 0, 40, 5)
        CODE_GENDER       = st.selectbox("Gender", [0.07, 0.10], format_func=lambda x: "Female" if x == 0.07 else "Male")
        NAME_EDUCATION_TYPE = st.selectbox("Education", [0.05, 0.07, 0.09, 0.11],
                              format_func=lambda x: {0.05:"Higher Education", 0.07:"Secondary", 0.09:"Incomplete Higher", 0.11:"Lower Secondary"}[x])
        OWN_CAR_AGE       = st.slider("Car Age (years, 0 if no car)", 0, 50, 0)
        DAYS_ID_PUBLISH   = st.slider("Years since ID issued", 0, 20, 5)

    with col3:
        st.subheader("Credit History")
        prev_refusal_rate     = st.slider("Previous Refusal Rate", 0.0, 1.0, 0.1, 0.01)
        inst_late_payment_rate = st.slider("Installment Late Payment Rate", 0.0, 1.0, 0.05, 0.01)
        cc_avg_utilization_6m  = st.slider("CC Utilization (6m avg)", 0.0, 1.0, 0.3, 0.01)
        bur_active_ratio       = st.slider("Bureau Active Ratio", 0.0, 1.0, 0.3, 0.01)
        bur_avg_days_credit    = st.number_input("Bureau Avg Days Credit", -3000, 0, -500)
        REGION_RATING_CLIENT_W_CITY = st.selectbox("Region Rating", [1, 2, 3])

    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        input_data = {
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_ANNUITY': AMT_ANNUITY,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'AGE_YEARS': AGE_YEARS,
            'YEARS_EMPLOYED': YEARS_EMPLOYED,
            'CODE_GENDER': CODE_GENDER,
            'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
            'OWN_CAR_AGE': OWN_CAR_AGE,
            'DAYS_ID_PUBLISH': -DAYS_ID_PUBLISH * 365,
            'prev_refusal_rate': prev_refusal_rate,
            'inst_late_payment_rate': inst_late_payment_rate,
            'cc_avg_utilization_6m': cc_avg_utilization_6m,
            'bur_active_ratio': bur_active_ratio,
            'bur_avg_days_credit': bur_avg_days_credit,
            'REGION_RATING_CLIENT_W_CITY': REGION_RATING_CLIENT_W_CITY,
            'CNT_FAM_MEMBERS': 2,
        }

        all_features = list(set(feature_sets['full'] + feature_sets['lr_only'] + NEW_FEATURES))
        for feat in all_features:
            if feat not in input_data:
                input_data[feat] = 0

        input_df = pd.DataFrame([input_data])

        with st.spinner("Computing risk assessment..."):
            prob_xgb, prob_lr, prob_ann, score, X_xgb = predict_all(input_df)

        st.markdown("---")
        st.subheader("Risk Assessment Results")

        m1, m2, m3, m4 = st.columns(4)
        emoji, label = risk_color(prob_xgb)
        m1.metric("XGBoost", f"{prob_xgb:.1%}", label)
        m2.metric("Logistic Regression", f"{prob_lr:.1%}")
        m3.metric("ANN", f"{prob_ann:.1%}")
        m4.metric("Credit Score", f"{score}", "Higher = Lower Risk")

        st.markdown(f"### {emoji} Overall Assessment: **{label}**")

        st.subheader("Feature Contributions (SHAP)")
        shap_vals = explainer.shap_values(X_xgb)
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_xgb.iloc[0],
                feature_names=X_xgb.columns.tolist()
            ),
            max_display=12,
            show=False
        )
        st.pyplot(fig)
        plt.close()


# ── Page 2: Model Info ─────────────────────────────────────────────────────────
elif page == "Model Info":
    st.title("Model Performance Summary")

    results = [
        {'Model': 'XGBoost',             'AUC': 0.7866, 'KS': 0.4352, 'Gini': 0.5732, 'Recall': 0.6824},
        {'Model': 'ANN',                 'AUC': 0.7712, 'KS': 0.4149, 'Gini': 0.5424, 'Recall': 0.4149},
        {'Model': 'Logistic Regression', 'AUC': 0.7609, 'KS': 0.3901, 'Gini': 0.5218, 'Recall': 0.6860},
        {'Model': 'SVM',                 'AUC': 0.7603, 'KS': 0.3968, 'Gini': 0.5205, 'Recall': 0.2465},
    ]

    df_results = pd.DataFrame(results)
    st.dataframe(df_results.style.highlight_max(
        subset=['AUC','KS','Gini','Recall'],color='#1B4D3E'
    ), use_container_width=True)

    st.subheader("Top 15 Features (SHAP)")
    shap_data = [
        {'Feature': 'EXT_SOURCES_MEAN',          'Mean |SHAP|': 0.392},
        {'Feature': 'LOAN_PAYMENT_LENGTH',        'Mean |SHAP|': 0.159},
        {'Feature': 'pos_avg_instalment_future',  'Mean |SHAP|': 0.145},
        {'Feature': 'EXT_SOURCE_2x3',             'Mean |SHAP|': 0.132},
        {'Feature': 'CODE_GENDER',                'Mean |SHAP|': 0.121},
        {'Feature': 'pos_total_months',           'Mean |SHAP|': 0.116},
        {'Feature': 'prev_credit_to_app_ratio',   'Mean |SHAP|': 0.094},
        {'Feature': 'YEARS_EMPLOYED',             'Mean |SHAP|': 0.091},
        {'Feature': 'inst_late_payment_rate',     'Mean |SHAP|': 0.086},
        {'Feature': 'OWN_CAR_AGE',               'Mean |SHAP|': 0.083},
        {'Feature': 'NAME_EDUCATION_TYPE',        'Mean |SHAP|': 0.083},
        {'Feature': 'ORGANIZATION_TYPE',          'Mean |SHAP|': 0.083},
        {'Feature': 'AMT_ANNUITY',               'Mean |SHAP|': 0.080},
        {'Feature': 'NAME_FAMILY_STATUS',         'Mean |SHAP|': 0.078},
        {'Feature': 'bur_most_recent_credit',     'Mean |SHAP|': 0.075},
    ]
    st.dataframe(pd.DataFrame(shap_data), use_container_width=True)

    # Convert the static report asset queries into absolute paths to prevent cloud asset drops
    st.subheader("ROC Curves")
    roc_path = PROJECT_ROOT / "reports" / "roc_curves.png"
    if roc_path.exists():
        st.image(str(roc_path))

    st.subheader("SHAP Feature Importance")
    shap_img_path = PROJECT_ROOT / "reports" / "shap" / "shap_beeswarm.png"
    if shap_img_path.exists():
        st.image(str(shap_img_path))
