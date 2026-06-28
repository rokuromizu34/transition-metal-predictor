"""
Compare baseline v1 vs v2 performance
"""
from baseline import predict_lambda_max_cft as predict_v1
from baseline_v2 import predict_lambda_max_v2 as predict_v2

test_complexes = [
    ('[Co(NH3)6]3+', 'Co', 3, 'octahedral', 'NH3 (x6)', 475),
    ('[Ni(H2O)6]2+', 'Ni', 2, 'octahedral', 'H2O (x6)', 720),
    ('[Cu(H2O)6]2+', 'Cu', 2, 'octahedral', 'H2O (x6)', 810),
    ('[CoCl4]2-', 'Co', 2, 'tetrahedral', 'Cl- (x4)', 690),
    ('[CuCl4]2-', 'Cu', 2, 'tetrahedral', 'Cl- (x4)', 430),
    ('[Ni(NH3)6]2+', 'Ni', 2, 'octahedral', 'NH3 (x6)', 570),
    ('[Fe(H2O)6]2+', 'Fe', 2, 'octahedral', 'H2O (x6)', 305),
]

print("=" * 80)
print("BASELINE v1 vs v2 COMPARISON")
print("=" * 80)
print(f"{'Complex':<20} {'Actual':<8} {'v1 Pred':<8} {'v1 Err':<8} {'v2 Pred':<8} {'v2 Err':<8}")
print("-" * 80)

errors_v1 = []
errors_v2 = []

for name, metal, ox, geom, ligs, actual in test_complexes:
    pred_v1 = predict_v1(metal, ox, geom, ligs)
    pred_v2 = predict_v2(metal, ox, geom, ligs)
    
    err_v1 = abs(pred_v1 - actual)
    err_v2 = abs(pred_v2 - actual)
    
    errors_v1.append(err_v1)
    errors_v2.append(err_v2)
    
    print(f"{name[:18]:<20} {actual:<8.0f} {pred_v1:<8.0f} {err_v1:<8.0f} {pred_v2:<8.0f} {err_v2:<8.0f}")

mae_v1 = sum(errors_v1) / len(errors_v1)
mae_v2 = sum(errors_v2) / len(errors_v2)
improvement = mae_v1 - mae_v2
print("-" * 80)
print(f"{'SUMMARY':<20} {'':8} {'MAE v1':<8} {'':8} {'MAE v2':<8} {'Improv.':<8}")
print(f"{'Results':<20} {'':8} {mae_v1:<8.1f} {'':8} {mae_v2:<8.1f} {improvement:<8.1f}")
if mae_v2 < mae_v1:
    improvement_pct = (1 - mae_v2/mae_v1) * 100
    print(f"\n✅ v2 is BETTER by {improvement:.1f} nm ({improvement_pct:.1f}% improvement)")
else:
    print(f"\n❌ v2 is WORSE by {-improvement:.1f} nm")

print("=" * 80)