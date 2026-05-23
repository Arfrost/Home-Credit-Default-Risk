from src.config import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, TARGET
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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

DATA_PATH    = PROCESSED_DIR + 'df_model.parquet'
FEATURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_sets.json')

def add_features(df):
    # EXT_SOURCE interactions
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_STD']  = df[ext_cols].std(axis=1)
    df['EXT_SOURCES_MIN']  = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_1x2']   = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_2x3']   = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1x3']   = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']

    # Loan repayment length
    df['LOAN_PAYMENT_LENGTH'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)

    # Income per family member — only if column exists
    if 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)

    # Days employed anomaly flag
    if 'YEARS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOMALY'] = (df['YEARS_EMPLOYED'] == 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df

NEW_FEATURES = [
    'EXT_SOURCES_MEAN', 'EXT_SOURCES_STD', 'EXT_SOURCES_MIN',
    'EXT_SOURCE_1x2', 'EXT_SOURCE_2x3', 'EXT_SOURCE_1x3',
    'LOAN_PAYMENT_LENGTH', 'DAYS_EMPLOYED_ANOMALY'
    # INCOME_PER_PERSON removed — CNT_FAM_MEMBERS not in df_model
]
def load_data():
    df = pd.read_parquet(DATA_PATH)
    with open(FEATURE_PATH, 'r') as f:
        feature_sets = json.load(f)
    return df, feature_sets

def split_data(df, features):
    from sklearn.model_selection import train_test_split
    X = df[features].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}")
    print(f"Test default rate:  {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test

def get_tree_data():
    df, feature_sets = load_data()

    # Add new features
    df = add_features(df)

    # Extend full feature set with new features
    # Handle nulls for new features
    for col in NEW_FEATURES:
        if col in ['EXT_SOURCE_1x2', 'EXT_SOURCE_2x3', 'EXT_SOURCE_1x3',
                   'EXT_SOURCES_MEAN', 'EXT_SOURCES_STD', 'EXT_SOURCES_MIN']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    features = feature_sets['full'] + NEW_FEATURES

    X_train, X_test, y_train, y_test = split_data(df, features)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"Total features: {len(features)}")

    return X_train, X_test, y_train, y_test, scale_pos_weight

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scale_pos_weight = get_tree_data()
    print("Preprocessing complete.")

    