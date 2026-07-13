import joblib
import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
model = joblib.load(ROOT / "models" / "best_ml_model.pkl")

TEST_CASES = [
    ('Co', 3, 'NH3', 'octahedral', 475),
    ('Co', 2, 'H2O', 'octahedral', 510),
    ('Cu', 2, 'H2O', 'octahedral', 794),
    ('Ni', 2, 'H2O', 'octahedral', 720),
    ('Cr', 3, 'H2O', 'octahedral', 573),
    ('Ti', 3, 'H2O', 'octahedral', 493),
]

METAL_Z = {'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29}
D_ELECTRONS = {
    ('Ti',3):1,('V',3):2,('Cr',3):3,('Mn',2):5,('Mn',3):4,
    ('Fe',2):6,('Fe',3):5,('Co',2):7,('Co',3):6,
    ('Ni',2):8,('Cu',2):9,('Cu',1):10,
}
LIGAND_STRENGTH = {
    'H2O': 1.00, 'NH3': 1.25, 'en': 1.40, 'CN': 0.42,
    'Cl': 0.73, 'Br': 0.70, 'I': 0.68, 'F': 0.85,
}
GEOMETRY = {'octahedral':1.00,'tetrahedral':0.44,'square_planar':1.20}

def make_features(metal, ox, lig, geom):
    return {
        'metal_Z': METAL_Z.get(metal, 26),
        'ox_state': ox,
        'd_electrons': D_ELECTRONS.get((metal, ox), 5),
        'ligand_strength': LIGAND_STRENGTH.get(lig, 1.0),
        'coord_number': 6 if geom == 'octahedral' else 4,
        'geom_factor': GEOMETRY.get(geom, 1.0),
        'effective_field': LIGAND_STRENGTH.get(lig, 1.0) * GEOMETRY.get(geom, 1.0),
        'is_octahedral': int(geom == 'octahedral'),
        'is_tetrahedral': int(geom == 'tetrahedral'),
        'is_square_planar': int(geom == 'square_planar'),
    }

print(f"{'Complex':<20} {'Predicted':<12} {'Real':<8} {'Error':<10} {'Error %':<10}")
print("-" * 70)

errors = []
for metal, ox, lig, geom, real in TEST_CASES:
    feat = pd.DataFrame([make_features(metal, ox, lig, geom)])
    pred = float(model.predict(feat)[0])
    error = abs(pred - real)
    pct = (error / real) * 100
    errors.append(error)
    
    name = f"[{metal}({lig})]"
    print(f"{name:<20} {pred:<12.0f} {real:<8} {error:<10.0f} {pct:<10.1f}")

print("-" * 70)
print(f"MAE: {np.mean(errors):.1f} nm")