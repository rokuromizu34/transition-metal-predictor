"""
Проверка предсказаний модели против известных экспериментальных данных
"""
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
model = joblib.load(ROOT / "models" / "best_ml_model.pkl")
feature_names = list(joblib.load(ROOT / "models" / "feature_names.pkl"))
META_PATH = ROOT / "models" / "model_meta.pkl"
meta = joblib.load(META_PATH) if META_PATH.exists() else {}

# Проверочные случаи из учебников / литературы (НЕ из твоего train датасета!)
TEST_CASES = [
    # (metal, ox, ligand, geometry, real_lambda_max, source)
    ('Co', 3, 'NH3', 'octahedral', 475, 'Miessler Inorganic Chemistry'),
    ('Co', 2, 'H2O', 'octahedral', 510, 'Housecroft Inorganic Chemistry'),
    ('Cu', 2, 'H2O', 'octahedral', 794, 'Shriver & Atkins'),
    ('Ni', 2, 'H2O', 'octahedral', 720, 'Miessler'),
    ('Cr', 3, 'H2O', 'octahedral', 573, 'Housecroft'),
    ('Ti', 3, 'H2O', 'octahedral', 493, 'Shriver & Atkins'),
    ('Fe', 3, 'H2O', 'octahedral', 240, 'Miessler'),
]

GEOMETRY_CN     = {'octahedral':6,'tetrahedral':4,'square_planar':4}
GEOMETRY_FACTOR = {'octahedral':1.00,'tetrahedral':0.44,'square_planar':1.20}
METAL_Z         = {'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29}
D_ELECTRONS     = {
    ('Ti',3):1,('Ti',2):2,('V',3):2,('V',2):3,
    ('Cr',3):3,('Cr',2):4,('Mn',2):5,('Mn',3):4,
    ('Fe',2):6,('Fe',3):5,('Co',2):7,('Co',3):6,
    ('Ni',2):8,('Ni',3):7,('Cu',2):9,('Cu',1):10,
}
LIGAND_STRENGTH = {
    'H2O': 1.00, 'NH3': 1.25,
}

def build_features(metal, ox, ligand, geometry):
    gf = GEOMETRY_FACTOR.get(geometry, 1.0)
    lig = LIGAND_STRENGTH.get(ligand, 1.0)
    f = {
        'metal_Z': METAL_Z.get(metal, 26),
        'ox_state': ox,
        'd_electrons': D_ELECTRONS.get((metal, ox), 5),
        'ligand_strength': lig,
        'coord_number': GEOMETRY_CN.get(geometry, 6),
        'geom_factor': gf,
        'effective_field': lig * gf,
        'is_octahedral': int(geometry == 'octahedral'),
        'is_tetrahedral': int(geometry == 'tetrahedral'),
        'is_square_planar': int(geometry == 'square_planar'),
    }
    return pd.DataFrame([f]).reindex(columns=feature_names)

print(f"{'Complex':<20} {'Predicted':<12} {'Real':<8} {'Error':<10} {'Error %':<10} Source")
print("-" * 90)

errors = []
for metal, ox, lig, geom, real_lam, source in TEST_CASES:
    X = build_features(metal, ox, lig, geom)
    pred = float(model.predict(X)[0])
    if meta.get("target") == "wavenumber_cm-1":
        pred = 1e7 / pred  # перевод обратно в nm
    error = abs(pred - real_lam)
    error_pct = (error / real_lam) * 100
    errors.append(error)

    complex_name = f"[{metal}({lig})]{ox}+"
    print(f"{complex_name:<20} {pred:<12.0f} {real_lam:<8} {error:<10.0f} {error_pct:<10.1f} {source}")

print("-" * 90)
print(f"Mean Absolute Error on independent test set: {sum(errors)/len(errors):.1f} nm")
print(f"Max error: {max(errors):.1f} nm")
print(f"Min error: {min(errors):.1f} nm")