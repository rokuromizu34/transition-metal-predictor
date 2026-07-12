import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).parent.parent

raw = pd.read_csv(ROOT / "data/raw/complexes_raw.csv")
X   = pd.read_csv(ROOT / "data/processed/features.csv")
y_nm = pd.read_csv(ROOT / "data/processed/target.csv").iloc[:,0].astype(float)
y_cm = 1e7 / y_nm

model = joblib.load(ROOT / "models/best_ml_model.pkl")
meta_path = ROOT / "models/model_meta.pkl"
meta = joblib.load(meta_path) if meta_path.exists() else {}

groups = raw["source"]  # группируем по источнику

cv = GroupKFold(n_splits=5)
pred = cross_val_predict(model, X, y_cm, cv=cv, groups=groups, n_jobs=-1)

pred_nm = 1e7 / pred if meta.get("target") == "wavenumber_cm-1" else pred

mae = mean_absolute_error(y_nm, pred_nm)
print(f"GroupKFold(by source) MAE (nm): {mae:.1f}")