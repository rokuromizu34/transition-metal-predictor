"""
Leave-one-metal-out validation
Run from project root: python models/evaluate_metal_holdout.py

WHY THIS METRIC EXISTS
-----------------------
retrain.py reports a plain 5-fold KFold(shuffle=True) MAE. That CV
shuffles all 93 complexes together and randomly splits them into folds,
so almost every fold contains examples of almost every metal (Co, Ni,
Cu, Fe, Cr, Mn, V, Ti). The model is therefore always being tested on
metals it has already seen elsewhere in the training set - it only has
to interpolate between complexes of a *metal it already knows*, using
leftover ligand/geometry variation. That is an easy task and inflates
the apparent accuracy.

evaluate_groupcv.py improves on this by grouping folds by literature
`source`, which prevents leakage of complexes copied within the same
book/paper into both train and test - but every metal is still present
in both the train and test folds each time, since all complexes here
come from a single source (Miessler_Tarr_2014). So even that CV never
asks the question a real deployment will ask: "the app is given a
transition metal complex whose *metal* was never in the training data
at all - can it still predict lambda_max reasonably?"

This script answers exactly that question. Using GroupKFold with
groups = raw["metal"], and n_splits equal to the number of unique
metals (8), every metal is held out completely in exactly one fold:
the model is trained on the other 7 metals and asked to predict the
8th metal's complexes cold. This is a much harder, and much more
honest, test of the model's ability to generalize to genuinely new
transition metals rather than just interpolate within metals it
already knows - which is the real-world use case for this predictor.

NOTE ON THE FITTED MODEL: cross_val_predict() ignores any weights on
a pre-fitted estimator passed to it - it internally clones the
estimator (keeping only its hyperparameters) and refits a fresh copy
on each fold's training data. So there is no leakage from reusing the
hyperparameters of the production ExtraTreesRegressor (same settings
as models/retrain.py) here.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).parent.parent

# ── Load data (same as evaluate_groupcv.py) ─────────────────────────
raw = pd.read_csv(ROOT / "data/raw/complexes_raw.csv")
X = pd.read_csv(ROOT / "data/processed/features.csv")
y_nm = pd.read_csv(ROOT / "data/processed/target.csv").iloc[:, 0].astype(float)

groups = raw["metal"]  # leave-one-metal-out, NOT source
metals = sorted(groups.unique())
n_metals = len(metals)

print(f"Dataset: {len(raw)} complexes, {n_metals} unique metals: {metals}")

# ── Model: same hyperparameters as models/retrain.py ────────────────
# (a fresh estimator, since cross_val_predict clones/refits per fold
# regardless - see note above)
meta_path = ROOT / "models/model_meta.pkl"
meta = joblib.load(meta_path) if meta_path.exists() else {}

model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

# ── GroupKFold with groups = metal, one fold per metal ───────────────
cv = GroupKFold(n_splits=n_metals)
pred_nm = cross_val_predict(model, X, y_nm, cv=cv, groups=groups, n_jobs=-1)

if meta.get("target") == "wavenumber_cm-1":
    # defensive: this project's target.csv is lambda_max in nm, so this
    # branch should not trigger, but kept for consistency with
    # evaluate_groupcv.py in case the pipeline's target unit changes
    pred_nm = 1e7 / pred_nm

# ── Per-metal MAE ─────────────────────────────────────────────────────
results = []
print("\nLeave-one-metal-out results:")
print(f"{'Metal':<8} {'n_test':>8} {'MAE (nm)':>10}")
print("-" * 28)
for metal in metals:
    mask = (groups == metal).values
    mae_metal = mean_absolute_error(y_nm[mask], pred_nm[mask])
    n_test = int(mask.sum())
    results.append({"metal": metal, "n_test_samples": n_test, "mae": round(mae_metal, 1)})
    print(f"{metal:<8} {n_test:>8} {mae_metal:>10.1f}")

results_df = pd.DataFrame(results)

overall_mae = mean_absolute_error(y_nm, pred_nm)
print("-" * 28)
print(f"Overall leave-one-metal-out MAE (nm): {overall_mae:.1f}")

if "mae_kfold" in meta:
    print(f"(for comparison, retrain.py's KFold MAE: {meta['mae_kfold']} nm)")

# ── Save results CSV ──────────────────────────────────────────────────
out_csv = ROOT / "data/processed/metal_holdout_results.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved per-metal results to {out_csv}")

# ── Barplot ─────────────────────────────────────────────────────────
plot_df = results_df.sort_values("mae", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(plot_df["metal"], plot_df["mae"], color="#4C72B0")
ax.set_xlabel("Metal held out")
ax.set_ylabel("MAE (nm)")
ax.set_title("Leave-one-metal-out MAE by metal")
ax.axhline(overall_mae, color="red", linestyle="--", linewidth=1, label=f"Overall MAE = {overall_mae:.1f} nm")
for bar, n in zip(bars, plot_df["n_test_samples"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
            ha="center", va="bottom", fontsize=8)
ax.legend()
fig.tight_layout()

out_png = ROOT / "data/processed/metal_holdout_plot.png"
fig.savefig(out_png, dpi=150)
print(f"Saved plot to {out_png}")
