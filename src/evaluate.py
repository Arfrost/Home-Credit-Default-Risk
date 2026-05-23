from src.train_tree import CreditRiskANN
import sys
import numpy as np
import pandas as pd
import json
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.metrics import f1_score
import __main__
__main__.CreditRiskANN = CreditRiskANN
from src.preprocessing import get_tree_data,  add_features, NEW_FEATURES
from src.train_lr_svm import get_lr_svm_data
from src.config import RANDOM_STATE
import os
os.makedirs('reports', exist_ok=True)

# ── Core Metrics ───────────────────────────────────────────────────────────────
def compute_ks(y_true, y_prob):
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['cum_pos'] = (df['y']==1).cumsum() / (df['y']==1).sum()
    df['cum_neg'] = (df['y']==0).cumsum() / (df['y']==0).sum()
    return (df['cum_pos'] - df['cum_neg']).abs().max()

def compute_gini(auc):
    return 2 * auc - 1

def full_metrics(y_true, y_prob, model_name, threshold=0.5):
    auc  = roc_auc_score(y_true, y_prob)
    ks   = compute_ks(y_true, y_prob)
    gini = compute_gini(auc)
    y_pred = (y_prob >= threshold).astype(int)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  KS Stat   : {ks:.4f}")
    print(f"  Gini      : {gini:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")

    return {
        'model': model_name, 'auc': auc, 'ks': ks, 'gini': gini,
        'precision': precision, 'recall': recall, 'f1': f1,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

# ── DeLong Test ────────────────────────────────────────────────────────────────
def delong_test(y_true, y_prob1, y_prob2, model1_name, model2_name):
    """
    Non-parametric test for comparing two AUC values.
    H0: AUC1 == AUC2
    """
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N)
        T2[J] = T + 1
        return T2

    def fastDeLong(y_true, y_prob1, y_prob2):
        n1 = int(y_true.sum())
        n2 = len(y_true) - n1
        
        pos1 = y_prob1[y_true == 1]
        neg1 = y_prob1[y_true == 0]
        pos2 = y_prob2[y_true == 1]
        neg2 = y_prob2[y_true == 0]

        def auc_and_var(pos, neg):
            m, n = len(pos), len(neg)
            V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
            V01 = np.array([np.mean(p < pos) + 0.5 * np.mean(p == pos) for p in neg])
            auc = V10.mean()
            s10 = np.var(V10, ddof=1) / m
            s01 = np.var(V01, ddof=1) / n
            return auc, s10, s01, V10, V01

        auc1, s10_1, s01_1, V10_1, V01_1 = auc_and_var(pos1, neg1)
        auc2, s10_2, s01_2, V10_2, V01_2 = auc_and_var(pos2, neg2)

        cov10 = np.cov(V10_1, V10_2, ddof=1)[0,1] / n1
        cov01 = np.cov(V01_1, V01_2, ddof=1)[0,1] / n2

        var  = s10_1 + s01_1 + s10_2 + s01_2 - 2*cov10 - 2*cov01
        diff = auc1 - auc2
        z    = diff / np.sqrt(var) if var > 0 else 0
        p    = 2 * (1 - norm.cdf(abs(z)))
        return auc1, auc2, z, p

    y_true  = np.array(y_true)
    y_prob1 = np.array(y_prob1)
    y_prob2 = np.array(y_prob2)

    auc1, auc2, z, p = fastDeLong(y_true, y_prob1, y_prob2)

    print(f"\nDeLong Test: {model1_name} vs {model2_name}")
    print(f"  AUC {model1_name}: {auc1:.4f}")
    print(f"  AUC {model2_name}: {auc2:.4f}")
    print(f"  Z-statistic : {z:.4f}")
    print(f"  P-value     : {p:.4f}")
    print(f"  Significant : {'Yes' if p < 0.05 else 'No'} (α=0.05)")

    return {
        'model1': model1_name, 
        'model2': model2_name,
        'auc1': auc1, 
        'auc2': auc2, 
        'z': float(z), 
        'p': float(p),
        'significant': bool(p < 0.05)  # <-- Explicitly convert to standard Python bool
    }

# ── ROC Curve Plot ─────────────────────────────────────────────────────────────
def plot_roc_curves(y_test, predictions_dict):
    """predictions_dict = {'ModelName': y_prob_array}"""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#E84C4C', '#4C9BE8', '#2ECC71', '#F39C12']

    for (name, y_prob), color in zip(predictions_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')

    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: reports/roc_curves.png")

# ── KS Plot ────────────────────────────────────────────────────────────────────
def plot_ks_curve(y_true, y_prob, model_name):
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['cum_pos'] = (df['y']==1).cumsum() / (df['y']==1).sum()
    df['cum_neg'] = (df['y']==0).cumsum() / (df['y']==0).sum()
    ks_idx = (df['cum_pos'] - df['cum_neg']).abs().idxmax()
    ks_val = (df['cum_pos'] - df['cum_neg']).abs().max()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, len(df))
    ax.plot(x, df['cum_pos'].values, color='#E84C4C', lw=2, label='Default (1)')
    ax.plot(x, df['cum_neg'].values, color='#4C9BE8', lw=2, label='Non-Default (0)')
    ax.axvline(x=ks_idx/len(df), color='gray', linestyle='--', lw=1)
    ax.annotate(f'KS={ks_val:.4f}', xy=(ks_idx/len(df), 0.5), fontsize=12,
                color='black', fontweight='bold')
    ax.set_title(f'KS Curve — {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Population %')
    ax.set_ylabel('Cumulative %')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/ks_curve_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: reports/ks_curve_{model_name}.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load test data
    X_train, X_test, y_train, y_test, spw = get_tree_data()
    (X_train_lr, X_test_lr, y_train_lr, y_test_lr, scaler_lr,
     X_train_svm, X_test_svm, y_train_svm, scaler_svm) = get_lr_svm_data()

    # Load models
    xgb_model = joblib.load('models/xgboost.joblib')
    ann_bundle = joblib.load('models/ann.joblib')
    lr_bundle  = joblib.load('models/logistic_regression.joblib')
    svm_bundle = joblib.load('models/svm.joblib')

    # Get predictions
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_prob_lr  = lr_bundle['model'].predict_proba(X_test_lr)[:, 1]
    y_prob_svm = svm_bundle['model'].predict_proba(X_test_svm)[:, 1]

    # ANN
    ann_model  = ann_bundle['model']
    ann_scaler = ann_bundle['scaler']
    X_test_ann = torch.FloatTensor(ann_scaler.transform(X_test))
    ann_model.eval()
    with torch.no_grad():
        y_prob_ann = ann_model(X_test_ann).squeeze().numpy()

    # Full metrics
    from sklearn.metrics import f1_score
    results = []
    results.append(full_metrics(y_test, y_prob_xgb, 'XGBoost'))
    thresholds = np.arange(0.1, 0.5, 0.01)
    f1_scores = [f1_score(y_test, (y_prob_ann >= t).astype(int)) for t in thresholds]
    best_threshold_ann = thresholds[np.argmax(f1_scores)]
    print(f"Best ANN threshold: {best_threshold_ann:.2f}")
    results.append(full_metrics(y_test, y_prob_ann, 'ANN', threshold=best_threshold_ann))
    results.append(full_metrics(y_test_lr, y_prob_lr, 'LogisticRegression'))
    results.append(full_metrics(y_test, y_prob_svm, 'SVM'))

    # DeLong tests — XGBoost vs all
    delong_results = []
    delong_results.append(delong_test(y_test, y_prob_xgb, y_prob_ann, 'XGBoost', 'ANN'))
    delong_results.append(delong_test(y_test, y_prob_xgb, y_prob_lr, 'XGBoost', 'LR'))
    delong_results.append(delong_test(y_test, y_prob_xgb, y_prob_svm, 'XGBoost', 'SVM'))
    delong_results.append(delong_test(y_test, y_prob_ann, y_prob_lr, 'ANN', 'LR'))

    # ROC curves
    plot_roc_curves(y_test, {
        'XGBoost': y_prob_xgb,
        'ANN': y_prob_ann,
        'LR': y_prob_lr,
        'SVM': y_prob_svm
    })

    # KS curve for best model
    plot_ks_curve(y_test, y_prob_xgb, 'XGBoost')

    # Save all results
    with open('reports/full_results.json', 'w') as f:
        json.dump({'metrics': results, 'delong': delong_results}, f, indent=2)

    # Summary table
    print("\n===== FINAL COMPARISON TABLE =====")
    print(f"{'Model':20s} {'AUC':>8} {'KS':>8} {'Gini':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 64)
    for r in results:
        print(f"{r['model']:20s} {r['auc']:>8.4f} {r['ks']:>8.4f} {r['gini']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")