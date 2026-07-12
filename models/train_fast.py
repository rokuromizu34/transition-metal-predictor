import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent.parent
X = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
y_nm = pd.read_csv(ROOT / "data" / "processed" / "target.csv").iloc[:, 0].astype(float)

# учим не nm, а волновое число (cm^-1) — чаще лучше работает
y_cm = 1e7 / y_nm

model = ExtraTreesRegressor(
    n_estimators=1500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt",
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

pred_cm = cross_val_predict(model, X, y_cm, cv=cv, n_jobs=-1)
pred_nm = 1e7 / pred_cm
mae_nm = mean_absolute_error(y_nm, pred_nm)
print(f"CV MAE (nm): {mae_nm:.1f}")

model.fit(X, y_cm)

joblib.dump(model, ROOT / "models" / "best_ml_model.pkl")
joblib.dump(list(X.columns), ROOT / "models" / "feature_names.pkl")
joblib.dump({"target": "wavenumber_cm-1"}, ROOT / "models" / "model_meta.pkl")

print("Saved: models/best_ml_model.pkl")
print("Saved: models/feature_names.pkl")
print("Saved: models/model_meta.pkl")