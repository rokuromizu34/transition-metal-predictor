import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('data/raw/complexes_raw.csv')

DELTA_O = {
    ('Co', 2): 19608,
    ('Co', 3): 24390,
    ('Cr', 3): 17452,
    ('Cu', 2): 12594,
    ('Fe', 2): 10417,
    ('Mn', 2): 19048,
    ('Ni', 2): 13889,
    ('Ti', 3): 20284,
    ('V', 3): 17241,
}
LIGAND_STRENGTH = {
    'CN': 0.42,
    'NO2': 0.60,
    'dmg': 0.60,
    'dtc': 0.64,
    'I': 0.68,
    'Br': 0.70,
    'Cl': 0.73,
    'SCN': 0.75,
    'NCS': 0.77,
    'ox': 0.80,
    'F': 0.85,
    'acac': 0.86,
    'edta': 0.88,
    'H2O': 1.00,
    'dien': 1.02,
    'en': 1.04,
    'NH3': 1.10,
    'phen': 1.12,
    'bipy': 1.12,
}
GEOMETRY_FACTOR = {
    'octahedral': 1.00,
    'tetrahedral': 0.44,
    'square_planar': 1.25,
}

predictions = []
actuals = []

for idx, row in df.iterrows():
    metal = row['metal']
    ox = int(row['ox_state'])
    lig = str(row['ligands'])
    geom = row['geometry']
    lam = float(row['lambda_max'])
    
    key = (metal, ox)
    if key not in DELTA_O:
        continue
    
    if '+' in lig:
        parts = lig.split('+')
        strength = np.mean([LIGAND_STRENGTH.get(p.strip(), 1.0) for p in parts])
    else:
        strength = LIGAND_STRENGTH.get(lig, 1.0)
    
    pred_lam = 1e7 / (DELTA_O[key] * strength)
    predictions.append(pred_lam)
    actuals.append(lam)

predictions = np.array(predictions)
actuals = np.array(actuals)

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
ss_res = np.sum((actuals - predictions) ** 2)
ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*50)
print("CFT BASELINE MODEL")
print("="*50)
print(f"MAE:  {mae:.2f} nm")
print(f"RMSE: {rmse:.2f} nm")
print(f"R2:   {r2:.4f}")
print(f"N:    {len(predictions)}")
print("="*50 + "\n")
print("\nTEST PREDICTIONS:")
print("="*60)
if __name__ == "__main__":
    # ... весь код выше ...
    
    print("\n" + "="*60)
    print("SPOT CHECKS")
    print("="*60)
    
    test_cases = [
        ('Co', 3, 'NH3', 475),
        ('Co', 2, 'H2O', 510),
        ('Cu', 2, 'H2O', 794),
        ('Ni', 2, 'H2O', 720),
    ]
    
    for metal, ox, lig, actual in test_cases:
        key = (metal, ox)
        if key in DELTA_O:
            strength = LIGAND_STRENGTH.get(lig, 1.0)
            pred = 1e7 / (DELTA_O[key] * strength)
            error = abs(pred - actual)
            pct = (error / actual) * 100
            print(f"{metal}({lig}) pred={pred:.0f} actual={actual} error={error:.0f}nm ({pct:.1f}%)")
    
    print("="*60)