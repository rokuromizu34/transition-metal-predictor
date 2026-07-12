import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent.parent
X = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
y_nm = pd.read_csv(ROOT / "data" / "processed" / "target.csv").iloc[:, 0].astype(float)

y_cm = 1e7 / y_nm

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RF": RandomForestRegressor(random_state=42),
    "ET": ExtraTreesRegressor(random_state=42),
}

param_grid = {
    "RF": {
        "n_estimators": [500, 1000],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    },
    "ET": {
        "n_estimators": [500, 1000],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7, 1.0],
    },
}

best = None
best_name = None
best_mae_nm = 1e9

for name, model in models.items():
    gs = GridSearchCV(
        model,
        param_grid[name],
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X, y_cm)
    y_cm_pred = gs.best_estimator_.predict(X)
    y_nm_pred = 1e7 / y_cm_pred
    train_mae_nm = mean_absolute_error(y_nm, y_nm_pred)

    print("=" * 70)
    print(name, "best params:", gs.best_params_)
    print("CV MAE (in cm^-1):", -gs.best_score_)
    print("TRAIN MAE (in nm):", train_mae_nm)

    if train_mae_nm < best_mae_nm:
        best_mae_nm = train_mae_nm
        best = gs.best_estimator_
        best_name = name

print("=" * 70)
print("BEST:", best_name, "train MAE nm:", best_mae_nm)

joblib.dump(best, ROOT / "models" / "best_ml_model.pkl")
joblib.dump(X.columns, ROOT / "models" / "feature_names.pkl")

# сохраняем флаг, что модель предсказывает cm^-1
meta = {"target": "wavenumber_cm-1", "note": "Predicts 1e7/lambda_nm"}
joblib.dump(meta, ROOT / "models" / "model_meta.pkl")

print("Saved: best_ml_model.pkl, feature_names.pkl, model_meta.pkl")