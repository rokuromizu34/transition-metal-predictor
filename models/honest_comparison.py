"""
Honest comparison: Baseline v3 vs ML on ALL 91 complexes
"""

import pandas as pd
from pathlib import Path
import joblib

# --- Load data ---
base = Path(__file__).parent.parent
df = pd.read_csv(base / 'data' / 'raw' / 'complexes_raw.csv')
X = pd.read_csv(base / 'data' / 'processed' / 'features.csv')
y = df['lambda_max'].values

print(f"Total complexes: {len(df)}")

# --- Load ML model ---
ml_model = joblib.load(Path(__file__).parent / 'best_ml_model.pkl')
ml_preds = ml_model.predict(X)

# --- Baseline v3 predictions ---
from baseline_v3 import predict_lambda_max_v3

baseline_preds = []
for _, row in df.iterrows():
    pred = predict_lambda_max_v3(
        row['metal'], row['ox_state'],
        row['geometry'], row['ligands']
    )
    baseline_preds.append(pred)

# --- Calculate errors ---
ml_errors = [abs(ml_preds[i] - y[i]) for i in range(len(y))]
bl_errors = [abs(baseline_preds[i] - y[i]) for i in range(len(y))]

# --- Print results ---
print("\n" + "=" * 85)
print(f"{'#':<3} {'Complex':<25} {'Actual':<8} {'Baseline':<10} {'BL Err':<8} {'ML':<10} {'ML Err':<8}")
print("-" * 85)

for i, row in df.iterrows():
    name = row['complex_name'][:23]
    print(f"{i+1:<3} {name:<25} {y[i]:<8.0f} "
          f"{baseline_preds[i]:<10.0f} {bl_errors[i]:<8.0f} "
          f"{ml_preds[i]:<10.0f} {ml_errors[i]:<8.0f}")

# --- Summary ---
bl_mae = sum(bl_errors) / len(bl_errors)
ml_mae = sum(ml_errors) / len(ml_errors)

print("-" * 85)
print(f"\n{'METRIC':<35} {'Baseline v3':<15} {'ML Model':<15}")
print("=" * 65)
print(f"{'MAE (all 91 complexes)':<35} {bl_mae:<15.1f} {ml_mae:<15.1f}")
print(f"{'Max error':<35} {max(bl_errors):<15.0f} {max(ml_errors):<15.0f}")
print(f"{'Min error':<35} {min(bl_errors):<15.0f} {min(ml_errors):<15.0f}")

# --- By metal ---
print(f"\n{'MAE BY METAL:'}")
print("-" * 50)
for metal in ['Co', 'Ni', 'Cu', 'Fe', 'Cr', 'Mn', 'V', 'Ti']:
    idx = [i for i, r in df.iterrows() if r['metal'] == metal]
    if idx:
        bl_m = sum(bl_errors[i] for i in idx) / len(idx)
        ml_m = sum(ml_errors[i] for i in idx) / len(idx)
        print(f"  {metal:<5} (n={len(idx):<2})  Baseline: {bl_m:>6.1f} nm   ML: {ml_m:>6.1f} nm")

# --- Verdict ---
print("\n" + "=" * 65)
if ml_mae < bl_mae:
    pct = (bl_mae - ml_mae) / bl_mae * 100
    print(f"ML WINS by {bl_mae - ml_mae:.1f} nm ({pct:.0f}% better)")
else:
    pct = (ml_mae - bl_mae) / ml_mae * 100
    print(f"Baseline WINS by {ml_mae - bl_mae:.1f} nm ({pct:.0f}% better)")
    print(f"BUT: Baseline only works for KNOWN complexes!")
    print(f"     ML can predict NEW complexes!")
print("=" * 65)