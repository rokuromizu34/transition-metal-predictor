"""
Train ML models on 91 complexes
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

base = Path(__file__).parent.parent / 'data' / 'processed'
X = pd.read_csv(base / 'features.csv')
y = pd.read_csv(base / 'target.csv').iloc[:, 0]

print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Lambda range: {y.min()} - {y.max()} nm")

print("\n" + "=" * 60)
print("MODEL 1: RANDOM FOREST")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

rf_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
rf_mae = -rf_scores.mean()
print(f"Cross-val MAE: {rf_mae:.1f} nm")

rf.fit(X, y)
rf_train_mae = mean_absolute_error(y, rf.predict(X))
print(f"Train MAE: {rf_train_mae:.1f} nm")

print("\nFeature importance:")
for name, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:<20} {imp:.3f} {bar}")

print("\n" + "=" * 60)
print("MODEL 2: GRADIENT BOOSTING")
print("=" * 60)

gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    min_samples_leaf=2,
    random_state=42
)

gb_scores = cross_val_score(gb, X, y, cv=5, scoring='neg_mean_absolute_error')
gb_mae = -gb_scores.mean()
print(f"Cross-val MAE: {gb_mae:.1f} nm")

gb.fit(X, y)
gb_train_mae = mean_absolute_error(y, gb.predict(X))
print(f"Train MAE: {gb_train_mae:.1f} nm")

print("\nFeature importance:")
for name, imp in sorted(zip(X.columns, gb.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:<20} {imp:.3f} {bar}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\n{'Model':<25} {'CV MAE (nm)':<15} {'Train MAE (nm)'}")
print("-" * 55)
print(f"{'Random Forest':<25} {rf_mae:<15.1f} {rf_train_mae:.1f}")
print(f"{'Gradient Boosting':<25} {gb_mae:<15.1f} {gb_train_mae:.1f}")

best_name = 'Gradient Boosting' if gb_mae < rf_mae else 'Random Forest'
best_mae = min(gb_mae, rf_mae)
best_model = gb if gb_mae < rf_mae else rf

print(f"\nBest model: {best_name} (MAE = {best_mae:.1f} nm)")

import joblib
model_path = Path(__file__).parent / 'best_ml_model.pkl'
joblib.dump(best_model, model_path)
print(f"Saved to: {model_path}")

print("=" * 60)