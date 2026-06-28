"""
Feature engineering v2 - for full dataset (90 complexes)
"""

import pandas as pd
from pathlib import Path

LIGAND_STRENGTH = {
    'I':    0.60,
    'Br':   0.70,
    'Cl':   0.78,
    'SCN':  0.83,
    'NCS':  0.83,
    'F':    0.90,
    'H2O':  1.00,
    'ox':   1.10,
    'acac': 1.15,
    'edta': 1.20,
    'NH3':  1.25,
    'dien': 1.30,
    'en':   1.40,
    'bipy': 1.50,
    'phen': 1.50,
    'dmg':  1.55,
    'dtc':  1.20,
    'NO2':  1.60,
    'CN':   1.70,
    'CO':   1.80
}

GEOMETRY_CN = {
    'octahedral':      6,
    'tetrahedral':     4,
    'square_planar':   4
}

GEOMETRY_FACTOR = {
    'octahedral':      1.00,
    'tetrahedral':     0.44,
    'square_planar':   1.20
}

METAL_Z = {
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29
}

D_ELECTRONS = {
    ('Ti', 3): 1, ('Ti', 2): 2,
    ('V',  3): 2, ('V',  2): 3,
    ('Cr', 3): 3, ('Cr', 2): 4,
    ('Mn', 2): 5, ('Mn', 3): 4,
    ('Fe', 2): 6, ('Fe', 3): 5,
    ('Co', 2): 7, ('Co', 3): 6,
    ('Ni', 2): 8, ('Ni', 3): 7,
    ('Cu', 2): 9, ('Cu', 1): 10,
}


def parse_ligand_strength(ligand_str):
    """Parse ligand string, handle mixed ligands like NH3+Cl"""
    parts = ligand_str.split('+')
    values = []
    for part in parts:
        part = part.strip()
        val = LIGAND_STRENGTH.get(part, 1.0)
        values.append(val)
    return sum(values) / len(values)


def extract_features(df):
    """Convert raw data into numerical features"""
    rows = []

    for _, row in df.iterrows():
        metal = row['metal']
        ox = int(row['ox_state'])
        ligand = str(row['ligands'])
        geometry = str(row['geometry'])

        f = {}

        f['metal_Z'] = METAL_Z.get(metal, 26)
        f['ox_state'] = ox
        f['d_electrons'] = D_ELECTRONS.get((metal, ox), 5)
        f['ligand_strength'] = parse_ligand_strength(ligand)
        f['coord_number'] = GEOMETRY_CN.get(geometry, 6)
        f['geom_factor'] = GEOMETRY_FACTOR.get(geometry, 1.0)
        f['effective_field'] = f['ligand_strength'] * f['geom_factor']
        f['is_octahedral'] = 1 if geometry == 'octahedral' else 0
        f['is_tetrahedral'] = 1 if geometry == 'tetrahedral' else 0
        f['is_square_planar'] = 1 if geometry == 'square_planar' else 0

        rows.append(f)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'complexes_raw.csv'
    df = pd.read_csv(data_path)

    X = extract_features(df)
    y = df['lambda_max']

    print(f"Dataset: {len(df)} complexes")
    print(f"Features: {list(X.columns)}")
    print(f"Lambda max range: {y.min()} - {y.max()} nm")
    print(f"\nFirst 5 rows:")
    print(X.head())

    out = Path(__file__).parent.parent / 'data' / 'processed'
    out.mkdir(parents=True, exist_ok=True)

    X.to_csv(out / 'features.csv', index=False)
    y.to_csv(out / 'target.csv', index=False)

    print(f"\nSaved to {out}")