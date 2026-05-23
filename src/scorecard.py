import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.preprocessing import load_data, add_features, NEW_FEATURES
from src.config import RANDOM_STATE

os.makedirs('reports/scorecard', exist_ok=True)

# ── Load model and data ────────────────────────────────────────────────────────
import json
with open('src/feature_sets.json') as f:
    feature_sets = json.load(f)

lr_bundle = joblib.load('models/logistic_regression.joblib')
lr_model  = lr_bundle['model']
scaler    = lr_bundle['scaler']

df, _ = load_data()
df = add_features(df)

lr_new = [f for f in NEW_FEATURES if f not in 
          ['EXT_SOURCE_1x2', 'EXT_SOURCE_2x3', 'EXT_SOURCE_1x3']]
lr_features = feature_sets['lr_only'] + lr_new

for col in lr_new:
    df[col] = df[col].fillna(df[col].median() if col in 
              ['EXT_SOURCES_MEAN','EXT_SOURCES_STD','EXT_SOURCES_MIN'] else 0)

# ── Scorecard Parameters ───────────────────────────────────────────────────────
# Standard scorecard scaling: score = PDO * log(odds) + offset
# PDO = Points to Double Odds (typically 20)
# Base score = 600 at odds of 50:1
PDO    = 20
ODDS   = 50
OFFSET = 600
FACTOR = PDO / np.log(2)

# ── Build Scorecard ────────────────────────────────────────────────────────────
def build_scorecard(model, scaler, features, df, n_bins=10):
    """
    Convert LR coefficients to integer scorecard points.
    Each feature is binned and assigned points based on WoE and LR coefficient.
    """
    scorecard = []
    
    # Get scaled coefficients
    coef     = model.coef_[0]
    intercept = model.intercept_[0]
    
    for i, feature in enumerate(features):
        if feature not in df.columns:
            continue
            
        series = df[feature].dropna()
        
        # Bin the feature
        try:
            bins = pd.qcut(series, n_bins, duplicates='drop', retbins=True)[1]
        except Exception:
            continue
        
        df_temp = df[[feature, 'TARGET']].copy().dropna()
        df_temp['bin'] = pd.cut(df_temp[feature], bins=bins, include_lowest=True)
        
        grouped = df_temp.groupby('bin', observed=True)['TARGET'].agg(['sum','count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        total_events     = grouped['events'].sum()
        total_non_events = grouped['non_events'].sum()
        
        if total_events == 0 or total_non_events == 0:
            continue
            
        grouped['dist_events']     = grouped['events'] / total_events
        grouped['dist_non_events'] = grouped['non_events'] / total_non_events
        grouped['dist_events']     = grouped['dist_events'].replace(0, 0.0001)
        grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 0.0001)
        grouped['woe']             = np.log(grouped['dist_events'] / grouped['dist_non_events'])
        
        # Convert WoE to scorecard points
        # Points = -(coefficient * WoE + intercept/n_features) * Factor + Offset/n_features
        n_features = len(features)
        for bin_label, row in grouped.iterrows():
            points = -(coef[i] * row['woe'] + intercept/n_features) * FACTOR + OFFSET/n_features
            scorecard.append({
                'feature':      feature,
                'bin':          str(bin_label),
                'woe':          round(row['woe'], 4),
                'events':       int(row['events']),
                'non_events':   int(row['non_events']),
                'event_rate':   round(row['events']/row['total'], 4),
                'points':       round(points)
            })
    
    return pd.DataFrame(scorecard)

scorecard_df = build_scorecard(lr_model, scaler, lr_features, df)

# ── Save and Display ───────────────────────────────────────────────────────────
scorecard_df.to_csv('reports/scorecard/scorecard.csv', index=False)
print(f"Scorecard built: {len(scorecard_df)} bins across {scorecard_df['feature'].nunique()} features")
print("\nSample — Top features by score range:")

score_range = scorecard_df.groupby('feature')['points'].agg(
    min_points='min', max_points='max'
)
score_range['range'] = score_range['max_points'] - score_range['min_points']
print(score_range.sort_values('range', ascending=False).head(15).to_string())

# ── Score Distribution Plot ────────────────────────────────────────────────────
# Compute total score per applicant
X = df[lr_features].copy()
for col in lr_new:
    X[col] = X[col].fillna(0)

X_scaled = scaler.transform(X)
log_odds  = lr_model.intercept_[0] + X_scaled.dot(lr_model.coef_[0])
scores    = -FACTOR * log_odds + OFFSET  # negative sign flips direction

df_scores = pd.DataFrame({'score': scores, 'TARGET': df['TARGET'].values})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution by target
axes[0].hist(df_scores[df_scores['TARGET']==0]['score'], bins=50,
             alpha=0.6, color='#4C9BE8', label='Non-Default', density=True)
axes[0].hist(df_scores[df_scores['TARGET']==1]['score'], bins=50,
             alpha=0.6, color='#E84C4C', label='Default', density=True)
axes[0].set_title('Credit Score Distribution by Target', fontweight='bold')
axes[0].set_xlabel('Credit Score')
axes[0].set_ylabel('Density')
axes[0].legend()

# Default rate by score band
df_scores['score_band'] = pd.cut(df_scores['score'], bins=10)
band_dr = df_scores.groupby('score_band', observed=True)['TARGET'].mean()
axes[1].bar(range(len(band_dr)), band_dr.values, color='#E84C4C', alpha=0.8)
axes[1].set_xticks(range(len(band_dr)))
axes[1].set_xticklabels([str(b) for b in band_dr.index], rotation=45, ha='right', fontsize=7)
axes[1].set_title('Default Rate by Score Band', fontweight='bold')
axes[1].set_xlabel('Score Band')
axes[1].set_ylabel('Default Rate')
axes[1].axhline(df['TARGET'].mean(), color='black', linestyle='--', linewidth=1,
                label=f'Overall rate ({df["TARGET"].mean():.3f})')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/scorecard/score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: reports/scorecard/score_distribution.png")

# ── Score Statistics ───────────────────────────────────────────────────────────
print(f"\nScore Statistics:")
print(f"  Mean score (Non-Default): {df_scores[df_scores['TARGET']==0]['score'].mean():.1f}")
print(f"  Mean score (Default):     {df_scores[df_scores['TARGET']==1]['score'].mean():.1f}")
print(f"  Overall range:            {df_scores['score'].min():.0f} — {df_scores['score'].max():.0f}")
print("\nScorecard complete.")