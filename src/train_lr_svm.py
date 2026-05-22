import pandas as pd
import numpy as np
import json
import os
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.preprocessing import load_data, split_data, add_features, NEW_FEATURES
from src.config import PROCESSED_DIR, RANDOM_STATE

os.makedirs('models', exist_ok=True)

# ── Load and prepare data ──────────────────────────────────────────────────────
def get_lr_svm_data():
    df, feature_sets = load_data()
    df = add_features(df)

    # LR uses lr_only features + new features (excluding high VIF interactions)
    lr_new = [f for f in NEW_FEATURES if f not in 
              ['EXT_SOURCE_1x2', 'EXT_SOURCE_2x3', 'EXT_SOURCE_1x3']]
    
    # SVM uses reduced features + new features
    svm_new = NEW_FEATURES

    lr_features  = feature_sets['lr_only'] + lr_new
    svm_features = feature_sets['reduced'] + svm_new

    # Fill nulls for new features
    for col in NEW_FEATURES:
        if col in ['EXT_SOURCES_MEAN', 'EXT_SOURCES_STD', 'EXT_SOURCES_MIN',
                   'EXT_SOURCE_1x2', 'EXT_SOURCE_2x3', 'EXT_SOURCE_1x3']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    # Split
    X_train_lr, X_test_lr, y_train, y_test = split_data(df, lr_features)
    X_train_svm, X_test_svm, _, _          = split_data(df, svm_features)

    # Scale
    scaler_lr  = RobustScaler()
    scaler_svm = RobustScaler()

    X_train_lr_s  = scaler_lr.fit_transform(X_train_lr)
    X_test_lr_s   = scaler_lr.transform(X_test_lr)
    X_train_svm_s = scaler_svm.fit_transform(X_train_svm)
    X_test_svm_s  = scaler_svm.transform(X_test_svm)

    # SMOTE — only on training data
    smote = SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE, k_neighbors=5)

    X_train_lr_sm,  y_train_lr_sm  = smote.fit_resample(X_train_lr_s,  y_train)
    X_train_svm_sm, y_train_svm_sm = smote.fit_resample(X_train_svm_s, y_train)

    print(f"LR  — train: {X_train_lr_sm.shape},  features: {len(lr_features)}")
    print(f"SVM — train: {X_train_svm_sm.shape}, features: {len(svm_features)}")
    print(f"After SMOTE — default rate: {y_train_lr_sm.mean():.4f}")

    return (X_train_lr_sm, X_test_lr_s, y_train_lr_sm, y_test, scaler_lr,
            X_train_svm_sm, X_test_svm_s, y_train_svm_sm, scaler_svm)

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(y_test, y_prob, model_name):
    auc = roc_auc_score(y_test, y_prob)

    df_eval = pd.DataFrame({'y': y_test, 'prob': y_prob})
    df_eval = df_eval.sort_values('prob', ascending=False).reset_index(drop=True)
    df_eval['cum_pos'] = (df_eval['y'] == 1).cumsum() / (df_eval['y'] == 1).sum()
    df_eval['cum_neg'] = (df_eval['y'] == 0).cumsum() / (df_eval['y'] == 0).sum()
    ks   = (df_eval['cum_pos'] - df_eval['cum_neg']).abs().max()
    gini = 2 * auc - 1

    print(f"\n{'='*40}")
    print(f"{model_name} Results")
    print(f"{'='*40}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"KS Stat : {ks:.4f}")
    print(f"Gini    : {gini:.4f}")

    return {'model': model_name, 'auc': auc, 'ks': ks, 'gini': gini}

# ── 1. LOGISTIC REGRESSION ────────────────────────────────────────────────────
def objective_lr(trial, X_train, y_train, X_test, y_test):
    C         = trial.suggest_float('C', 1e-4, 10.0, log=True)
    penalty   = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver    = 'liblinear'  # supports both l1 and l2

    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def train_logistic_regression(X_train, X_test, y_train, y_test, scaler, n_trials=50):
    print("\nOptimizing Logistic Regression...")

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_lr(trial, X_train, y_train, X_test, y_test),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best = study.best_params
    model = LogisticRegression(
        C=best['C'], penalty=best['penalty'],
        solver='liblinear', class_weight='balanced',
        random_state=RANDOM_STATE, max_iter=1000
    )
    model.fit(X_train, y_train)

    # Save model + scaler together
    joblib.dump({'model': model, 'scaler': scaler}, 'models/logistic_regression.joblib')
    print(f"Best LR params: {best}")
    print(f"Best LR AUC: {study.best_value:.4f}")

    return model, best, study.best_value

# ── 2. SVM ────────────────────────────────────────────────────────────────────
def objective_svm(trial, X_train, y_train, X_test, y_test):
    C      = trial.suggest_float('C', 0.01, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear'])
    gamma  = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel == 'rbf' else 'scale'

    model = SVC(
        C=C, kernel=kernel, gamma=gamma,
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def train_svm(X_train, X_test, y_train, y_test, scaler, n_trials=10):
    print("\nOptimizing SVM...")
    print("Note: SVM running on subsample for speed...")

    # Subsample for SVM — too slow on full dataset
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), size=min(10000, len(X_train)), replace=False
    )
    X_train_sub = X_train[idx]
    y_train_sub = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_svm(trial, X_train_sub, y_train_sub, X_test, y_test),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best = study.best_params
    model = SVC(
        C=best['C'], kernel=best['kernel'],
        gamma=best.get('gamma', 'scale'),
        probability=True, class_weight='balanced',
        random_state=RANDOM_STATE
    )
    # Train final model on full subsample
    model.fit(X_train_sub, y_train_sub)

    joblib.dump({'model': model, 'scaler': scaler}, 'models/svm.joblib')
    print(f"Best SVM params: {best}")
    print(f"Best SVM AUC: {study.best_value:.4f}")

    return model, best, study.best_value

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    (X_train_lr, X_test_lr, y_train_lr, y_test,  scaler_lr,
     X_train_svm, X_test_svm, y_train_svm, scaler_svm) = get_lr_svm_data()

    results = []

    # Logistic Regression
    lr_model, lr_params, _ = train_logistic_regression(
        X_train_lr, X_test_lr, y_train_lr, y_test, scaler_lr, n_trials=50
    )
    y_prob_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    results.append(evaluate(y_test, y_prob_lr, 'LogisticRegression'))

    # SVM
    svm_model, svm_params, _ = train_svm(
        X_train_svm, X_test_svm, y_train_svm, y_test, scaler_svm, n_trials=10
    )
    y_prob_svm = svm_model.predict_proba(X_test_svm)[:, 1]
    results.append(evaluate(y_test, y_prob_svm, 'SVM'))

    # Save results
    with open('models/lr_svm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"{r['model']:20s} AUC={r['auc']:.4f}  KS={r['ks']:.4f}  Gini={r['gini']:.4f}")