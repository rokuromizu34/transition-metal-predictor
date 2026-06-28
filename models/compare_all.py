"""
Compare all three baseline versions
"""

from baseline import predict_lambda_max_cft as predict_v1
from baseline_v2 import predict_lambda_max_v2 as predict_v2
from baseline_v3 import predict_lambda_max_v3 as predict_v3

test_data = [
    ('[Co(NH3)6]3+', 'Co', 3, 'octahedral', 'NH3 (x6)', 475),
    ('[Ni(H2O)6]2+', 'Ni', 2, 'octahedral', 'H2O (x6)', 720),
    ('[Cu(H2O)6]2+', 'Cu', 2, 'octahedral', 'H2O (x6)', 810),
    ('[CoCl4]2-', 'Co', 2, 'tetrahedral', 'Cl- (x4)', 690),
    ('[CuCl4]2-', 'Cu', 2, 'tetrahedral', 'Cl- (x4)', 430),
    ('[Ni(NH3)6]2+', 'Ni', 2, 'octahedral', 'NH3 (x6)', 570),
    ('[Fe(H2O)6]2+', 'Fe', 2, 'octahedral', 'H2O (x6)', 305),
]

print("="*90)
print("COMPARISON: Baseline v1 vs v2 vs v3")
print("="*90)
print(f"{'Complex':<18} {'Actual':<7} {'v1':<7} {'Err1':<7} {'v2':<7} {'Err2':<7} {'v3':<7} {'Err3':<7}")
print("-"*90)

err1_list, err2_list, err3_list = [], [], []

for name, m, ox, geom, ligs, actual in test_data:
    p1 = predict_v1(m, ox, geom, ligs)
    p2 = predict_v2(m, ox, geom, ligs)
    p3 = predict_v3(m, ox, geom, ligs)
    
    e1, e2, e3 = abs(p1-actual), abs(p2-actual), abs(p3-actual)
    err1_list.append(e1)
    err2_list.append(e2)
    err3_list.append(e3)
    
    print(f"{name[:16]:<18} {actual:<7.0f} {p1:<7.0f} {e1:<7.0f} {p2:<7.0f} {e2:<7.0f} {p3:<7.0f} {e3:<7.0f}")

mae1 = sum(err1_list)/len(err1_list)
mae2 = sum(err2_list)/len(err2_list)
mae3 = sum(err3_list)/len(err3_list)

print("-"*90)
print(f"{'MAE':<18} {'':7} {mae1:<7.1f} {'':7} {mae2:<7.1f} {'':7} {mae3:<7.1f}")
print("="*90)

best = min(mae1, mae2, mae3)
if best == mae3:
    print("🏆 v3 (Empirical) is BEST!")
elif best == mae1:
    print("🏆 v1 (Simple CFT) is BEST!")
else:
    print("🏆 v2 (Improved CFT) is BEST!")