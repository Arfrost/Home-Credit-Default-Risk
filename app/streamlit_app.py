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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import add_features, NEW_FEATURES,CreditRiskANN
from src.config import MODEL_DIR

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide"
)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model  = joblib.load(f'{MODEL_DIR}xgboost.joblib')
    lr_bundle  = joblib.load(f'{MODEL_DIR}logistic_regression.joblib')
    ann_bundle = joblib.load(f'{MODEL_DIR}ann.joblib')
    with open('src/feature_sets.json') as f:
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
        # Build input row
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

        # Fill missing features with 0
        all_features = list(set(feature_sets['full'] + feature_sets['lr_only'] + NEW_FEATURES))
        for feat in all_features:
            if feat not in input_data:
                input_data[feat] = 0

        input_df = pd.DataFrame([input_data])

        with st.spinner("Computing risk assessment..."):
            prob_xgb, prob_lr, prob_ann, score, X_xgb = predict_all(input_df)

        # Results
        st.markdown("---")
        st.subheader("Risk Assessment Results")

        m1, m2, m3, m4 = st.columns(4)
        emoji, label = risk_color(prob_xgb)
        m1.metric("XGBoost", f"{prob_xgb:.1%}", label)
        m2.metric("Logistic Regression", f"{prob_lr:.1%}")
        m3.metric("ANN", f"{prob_ann:.1%}")
        m4.metric("Credit Score", f"{score}", "Higher = Lower Risk")

        st.markdown(f"### {emoji} Overall Assessment: **{label}**")

        # SHAP waterfall
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

# ── Page 2: Batch Prediction ───────────────────────────────────────────────────
elif page == "Batch Prediction":
    st.title("Batch Risk Assessment")
    st.markdown("Upload a CSV file to score multiple applicants at once.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_batch):,} applicants")
        st.dataframe(df_batch.head())

        if st.button("Score All Applicants", type="primary"):
            with st.spinner("Scoring..."):
                df_out = add_features(df_batch.copy())
                for col in NEW_FEATURES:
                    if col not in df_out.columns:
                        df_out[col] = 0
                    df_out[col] = df_out[col].fillna(0)

                all_features = feature_sets['full'] + NEW_FEATURES
                X = df_out.reindex(columns=all_features, fill_value=0)
                probs = xgb_model.predict_proba(X)[:, 1]

                df_batch['default_probability'] = probs
                df_batch['risk_label'] = pd.cut(
                    probs, bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )

                # Credit scores
                lr_new = [f for f in NEW_FEATURES if f not in
                          ['EXT_SOURCE_1x2','EXT_SOURCE_2x3','EXT_SOURCE_1x3']]
                X_lr   = df_out.reindex(columns=feature_sets['lr_only'] + lr_new, fill_value=0)
                X_lr_s = lr_bundle['scaler'].transform(X_lr)
                FACTOR = 20 / np.log(2)
                log_odds = lr_bundle['model'].intercept_[0] + X_lr_s.dot(lr_bundle['model'].coef_[0])
                df_batch['credit_score'] = (-FACTOR * log_odds + 600).astype(int)

            st.success(f"Scored {len(df_batch):,} applicants")
            st.dataframe(df_batch[['default_probability', 'risk_label', 'credit_score']].head(20))

            # Distribution
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(probs, bins=50, color='#4C9BE8', edgecolor='white')
            axes[0].set_title('Default Probability Distribution')
            axes[0].set_xlabel('Probability')
            axes[1].bar(['Low Risk', 'Medium Risk', 'High Risk'],
                       df_batch['risk_label'].value_counts()[['Low Risk','Medium Risk','High Risk']],
                       color=['#2ECC71','#F39C12','#E84C4C'])
            axes[1].set_title('Risk Category Distribution')
            st.pyplot(fig)
            plt.close()

            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("Download Scored CSV", csv,
                             "scored_applicants.csv", "text/csv")

# ── Page 3: Model Info ─────────────────────────────────────────────────────────
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
        subset=['AUC','KS','Gini','Recall'], color='#d4edda'
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

    st.subheader("ROC Curves")
    if os.path.exists('reports/roc_curves.png'):
        st.image('reports/roc_curves.png')

    st.subheader("SHAP Feature Importance")
    if os.path.exists('reports/shap/shap_beeswarm.png'):
        st.image('reports/shap/shap_beeswarm.png')