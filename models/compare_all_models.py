"""
Compare: Baseline v3 vs ML models
"""

import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from baseline_v3 import predict_lambda_max_v3
from feature_engineering import extract_features

print("="*80)
print("LOADING DATA AND MODELS")
print("="*80)

# Load data
data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'complexes_raw.csv'
df = pd.read_csv(data_path)

# Load ML model
ml_model = joblib.load('models/best_ml_model.pkl')

# Extract features
X = extract_features(df)

# ============================================================================
# PREDICTIONS
# ============================================================================

# Baseline v3
baseline_preds = df.apply(
    lambda row: predict_lambda_max_v3(
        row['metal'], row['ox_state'], 
        row['geometry'], row['ligands']
    ), axis=1
).values

# ML model
ml_preds = ml_model.predict(X)

# Actual
actual = df['lambda_max'].values

# Errors
err_baseline = abs(baseline_preds - actual)
err_ml = abs(ml_preds - actual)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*100)
print("DETAILED COMPARISON")
print("="*100)
print(f"{'#':<3} {'Complex':<18} {'Actual':<8} {'Baseline v3':<15} {'ML Model':<15}")
print(f"{'':3} {'':18} {'':8} {'(Err nm)':<15} {'(Err nm)':<15}")
print("-"*100)

for idx, row in df.iterrows():
    name = row['complex_name'][:16]
    actual_val = row['lambda_max']
    
    print(f"{idx+1:<3} {name:<18} {actual_val:<8.0f} "
          f"{baseline_preds[idx]:.0f} ({err_baseline[idx]:>5.1f})   "
          f"{ml_preds[idx]:.0f} ({err_ml[idx]:>5.1f})")

# ============================================================================
# STATISTICS
# ============================================================================

mae_baseline = err_baseline.mean()
mae_ml = err_ml.mean()
rmse_baseline = (err_baseline ** 2).mean() ** 0.5
rmse_ml = (err_ml ** 2).mean() ** 0.5
max_err_baseline = err_baseline.max()
max_err_ml = err_ml.max()
min_err_baseline = err_baseline.min()
min_err_ml = err_ml.min()

print("-"*100)
print(f"{'SUMMARY':<21} {'':8} {mae_baseline:<15.1f} {mae_ml:<15.1f}")
print("="*100)

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"\n{'Metric':<30} {'Baseline v3':<20} {'ML Model':<20}")
print("-"*80)
print(f"{'Mean Absolute Error':<30} {mae_baseline:<20.1f} {mae_ml:<20.1f}")
print(f"{'Root Mean Sq. Error':<30} {rmse_baseline:<20.1f} {rmse_ml:<20.1f}")
print(f"{'Max Error':<30} {max_err_baseline:<20.1f} {max_err_ml:<20.1f}")
print(f"{'Min Error':<30} {min_err_baseline:<20.1f} {min_err_ml:<20.1f}")

# ============================================================================
# VERDICT
# ============================================================================

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if mae_ml < mae_baseline:
    improvement = mae_baseline - mae_ml
    improvement_pct = (improvement / mae_baseline) * 100
    print(f"\n✅ ML Model is BETTER by {improvement:.1f} nm ({improvement_pct:.1f}% improvement)")
    print(f"   Use ML Model for predictions!")
else:
    print(f"\n⚠️  Baseline v3 is still better")
    print(f"   ML needs MORE DATA to train properly")
    print(f"   Baseline v3 works well on this small dataset")

print("\n" + "="*80)