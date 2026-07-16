"""
Retrain model on current dataset
Run from project root: python models/retrain.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (
    cross_val_score, GroupKFold
)
from sklearn.metrics import mean_absolute_error
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

# ── Load data ─────────────────────────────────────
print("Loading dataset...")
raw = pd.read_csv(ROOT / "data/raw/complexes_raw.csv")
X   = pd.read_csv(ROOT / "data/processed/features.csv")
y   = pd.read_csv(ROOT / "data/processed/target.csv")
y   = y.iloc[:, 0].astype(float)

print(f"Dataset: {len(raw)} complexes")
print(f"Metals: {sorted(raw['metal'].unique())}")
print(f"Geometries: {sorted(raw['geometry'].unique())}")
print(f"Lambda range: {y.min():.0f} - {y.max():.0f} nm")

# ── Train model ────────────────────────────────────
print("\nTraining ExtraTreesRegressor...")
model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# Cross-validation by source (honest evaluation)
groups = raw["source"]
cv     = GroupKFold(n_splits=5)

scores = cross_val_score(
    model, X, y,
    cv=cv,
    groups=groups,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print(f"\nGroupKFold CV Results:")
print(f"  MAE: {-scores.mean():.1f} ± {scores.std():.1f} nm")

# Train final model on all data
model.fit(X, y)

# ── Feature importance ─────────────────────────────
print("\nFeature Importance:")
names = list(X.columns)
for name, imp in sorted(
    zip(names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    bar = "█" * int(imp * 50)
    print(f"  {name:<20} {bar} {imp:.3f}")

# ── Save model ─────────────────────────────────────
print("\nSaving model...")
joblib.dump(model, ROOT / "models/best_ml_model.pkl")

# Update meta
meta = {
    "target": "nm",
    "n_complexes": len(raw),
    "mae_groupkfold": round(-scores.mean(), 1),
    "metals": sorted(raw["metal"].unique().tolist()),
    "model": "ExtraTreesRegressor",
    "n_estimators": 200
}
joblib.dump(meta, ROOT / "models/model_meta.pkl")

print(f"✅ Model saved!")
print(f"✅ Meta saved: {meta}")
print(f"\nDone! MAE = {-scores.mean():.1f} nm")