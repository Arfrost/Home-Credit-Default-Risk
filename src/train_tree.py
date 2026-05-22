import pandas as pd
import numpy as np
import json
import os
import joblib
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.preprocessing import get_tree_data
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# ── Paths ──────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
RESULTS_PATH = 'models/tree_results.json'

# ── Load data once ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, scale_pos_weight = get_tree_data()

# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    # KS Statistic
    df_eval = pd.DataFrame({'y': y_test, 'prob': y_prob})
    df_eval = df_eval.sort_values('prob', ascending=False).reset_index(drop=True)
    df_eval['cum_pos'] = (df_eval['y'] == 1).cumsum() / (df_eval['y'] == 1).sum()
    df_eval['cum_neg'] = (df_eval['y'] == 0).cumsum() / (df_eval['y'] == 0).sum()
    ks = (df_eval['cum_pos'] - df_eval['cum_neg']).abs().max()
    
    gini = 2 * auc - 1
    
    print(f"\n{'='*40}")
    print(f"{model_name} Results")
    print(f"{'='*40}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"KS Stat : {ks:.4f}")
    print(f"Gini    : {gini:.4f}")
    
    return {'model': model_name, 'auc': auc, 'ks': ks, 'gini': gini}

# ── 1. XGBOOST ─────────────────────────────────────────────────────────────────
def objective_xgb(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight':  scale_pos_weight,
        'eval_metric':       'auc',
        'random_state':      42,
        'n_jobs':            -1,
        'tree_method':       'hist'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def train_xgboost(n_trials=50):
    print("\nOptimizing XGBoost...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    })
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    joblib.dump(model, 'models/xgboost.joblib')
    print(f"Best XGBoost AUC: {study.best_value:.4f}")
    return model, study.best_params, study.best_value

# ── 3. ANN ─────────────────────────────────────────────────────────────────────
class CreditRiskANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

def train_ann_trial(trial, X_tr, y_tr, X_val, y_val, input_dim):
    # Hyperparameters
    n_layers     = trial.suggest_int('n_layers', 2, 4)
    hidden_dim   = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout      = trial.suggest_float('dropout', 0.1, 0.5)
    lr           = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [256, 512, 1024])
    
    hidden_layers = [hidden_dim] * n_layers
    
    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    # Class weight for imbalance
    pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32)
    
    # Tensors
    X_t = torch.FloatTensor(X_tr_s)
    y_t = torch.FloatTensor(y_tr.values).unsqueeze(1)
    X_v = torch.FloatTensor(X_val_s)
    y_v = torch.FloatTensor(y_val.values).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model     = CreditRiskANN(input_dim, hidden_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train for 20 epochs per trial — fast evaluation
    model.train()
    for epoch in range(20):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            # Manual pos_weight scaling
            weights = torch.where(y_batch == 1, pos_weight, torch.ones_like(y_batch))
            loss = (criterion(output, y_batch) * weights).mean()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_v).squeeze().numpy()
    
    return roc_auc_score(y_val, y_prob)

def train_ann(n_trials=30):
    print("\nOptimizing ANN...")
    input_dim = X_train.shape[1]
    
    def objective_ann(trial):
        return train_ann_trial(
            trial, X_train, y_train, X_test, y_test, input_dim
        )
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective_ann, n_trials=n_trials, show_progress_bar=True)
    
    # Retrain best model for more epochs
    best = study.best_params
    hidden_layers = [best['hidden_dim']] * best['n_layers']
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32)
    X_t = torch.FloatTensor(X_train_s)
    y_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=best['batch_size'], shuffle=True)
    
    model     = CreditRiskANN(input_dim, hidden_layers, best['dropout'])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'])
    
    model.train()
    for epoch in range(50):  # full training
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            weights = torch.where(y_batch == 1, pos_weight, torch.ones_like(y_batch))
            loss = (criterion(output, y_batch) * weights).mean()
            loss.backward()
            optimizer.step()
    
    # Save model and scaler together
    joblib.dump({'model': model, 'scaler': scaler}, 'models/ann.joblib')
    print(f"Best ANN AUC: {study.best_value:.4f}")
    return model, scaler, study.best_params, study.best_value

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = []
    
    # XGBoost
    xgb_model, xgb_params, xgb_auc = train_xgboost(n_trials=50)
    results.append(evaluate(xgb_model, X_test, y_test, 'XGBoost'))
    
    
    # ANN
    ann_model, ann_scaler, ann_params, ann_auc = train_ann(n_trials=30)
    
    # ANN evaluation
    scaler = ann_scaler
    X_test_s = scaler.transform(X_test)
    X_t = torch.FloatTensor(X_test_s)
    ann_model.eval()
    with torch.no_grad():
        y_prob_ann = ann_model(X_t).squeeze().numpy()
    
    ann_results = {
        'model': 'ANN',
        'auc': roc_auc_score(y_test, y_prob_ann),
        'ks': 0,  # compute inline
        'gini': 2 * roc_auc_score(y_test, y_prob_ann) - 1
    }
    results.append(ann_results)
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"{r['model']:15s} AUC={r['auc']:.4f}  KS={r['ks']:.4f}  Gini={r['gini']:.4f}")