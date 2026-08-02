"""
Compare model complexity under leave-one-metal-out validation.

Question: is ExtraTreesRegressor (200 trees) actually the best choice
for THIS validation scheme, given only 93 training complexes? Tree
ensembles can overfit small datasets; a simpler regularized linear
model might generalize better to a metal it has never seen, even if
it looks slightly worse under standard KFold.

Run from project root: python models/compare_models_holdout.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent

raw = pd.read_csv(ROOT / "data/raw/complexes_raw.csv")
X = pd.read_csv(ROOT / "data/processed/features.csv")
y = pd.read_csv(ROOT / "data/processed/target.csv").iloc[:, 0].astype(float)
groups = raw["metal"]
metals = sorted(groups.unique())
n_metals = len(metals)

models = {
    "ExtraTrees (200 trees)": ExtraTreesRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1
    ),
    "Ridge (alpha=1.0, scaled)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "LinearRegression (scaled)": make_pipeline(StandardScaler(), LinearRegression()),
}

kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
group_cv = GroupKFold(n_splits=n_metals)

print(f"{'Model':<28} {'KFold MAE':>12} {'Leave1MetalOut MAE':>20}")
print("-" * 62)

summary_rows = []
per_metal_rows = {}

for name, model in models.items():
    # standard KFold MAE (for comparison, mirrors retrain.py)
    kfold_scores = cross_val_score(model, X, y, cv=kfold_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    kfold_mae = -kfold_scores.mean()

    # leave-one-metal-out MAE
    pred = cross_val_predict(model, X, y, cv=group_cv, groups=groups, n_jobs=-1)
    holdout_mae = mean_absolute_error(y, pred)

    print(f"{name:<28} {kfold_mae:>12.1f} {holdout_mae:>20.1f}")
    summary_rows.append({"model": name, "kfold_mae": round(kfold_mae, 1), "holdout_mae": round(holdout_mae, 1)})

    metal_maes = {}
    for metal in metals:
        mask = (groups == metal).values
        metal_maes[metal] = round(mean_absolute_error(y[mask], pred[mask]), 1)
    per_metal_rows[name] = metal_maes

print("-" * 62)

# ── Per-metal comparison table ───────────────────────────────────────
print("\nPer-metal leave-one-metal-out MAE by model:")
header = f"{'Metal':<6}" + "".join(f"{name:>26}" for name in models)
print(header)
for metal in metals:
    row = f"{metal:<6}" + "".join(f"{per_metal_rows[name][metal]:>26.1f}" for name in models)
    print(row)

# ── Save results ──────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
out_csv = ROOT / "data/processed/model_comparison_holdout.csv"
summary_df.to_csv(out_csv, index=False)

per_metal_df = pd.DataFrame(per_metal_rows)
per_metal_df.index.name = "metal"
out_csv2 = ROOT / "data/processed/model_comparison_per_metal.csv"
per_metal_df.to_csv(out_csv2)

print(f"\nSaved summary to {out_csv}")
print(f"Saved per-metal breakdown to {out_csv2}")

best = summary_df.loc[summary_df["holdout_mae"].idxmin()]
print(f"\nBest generalizing model (lowest leave-one-metal-out MAE): {best['model']} ({best['holdout_mae']} nm)")
