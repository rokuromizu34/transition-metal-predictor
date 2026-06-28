"""
Baseline model using Crystal Field Theory (CFT)
Predicts lambda_max based on spectrochemical series and geometry
"""

import pandas as pd
import numpy as np
LIGAND_STRENGTH = {
    'I-': 0.6,
    'Br-': 0.7,
    'Cl-': 0.8,
    'F-': 0.9,
    'OH-': 1.0,
    'H2O': 1.0,   
    'NH3': 1.25,
    'en': 1.4,   
    'bpy': 1.5,   
    'phen': 1.5,   
    'NO2-': 1.6,
    'CN-': 1.7,
    'CO': 1.8
}

GEOMETRY_FACTOR = {
    'octahedral': 1.0,
    'tetrahedral': 0.45, 
    'square_planar': 1.2,
    'trigonal_bipyramidal': 0.85,
    'square_pyramidal': 0.9,
    'linear': 0.3
}
# d-electron count
D_ELECTRONS = {
    ('Sc', 3): 0, ('Ti', 2): 2, ('Ti', 3): 1, ('Ti', 4): 0,
    ('V', 2): 3, ('V', 3): 2, ('V', 4): 1, ('V', 5): 0,
    ('Cr', 2): 4, ('Cr', 3): 3, ('Cr', 6): 0,
    ('Mn', 2): 5, ('Mn', 3): 4, ('Mn', 4): 3, ('Mn', 7): 0,
    ('Fe', 2): 6, ('Fe', 3): 5,
    ('Co', 2): 7, ('Co', 3): 6,
    ('Ni', 2): 8, ('Ni', 3): 7,
    ('Cu', 1): 10, ('Cu', 2): 9,
    ('Zn', 2): 10
}


def extract_ligand_type(ligand_str):
    """Extract dominant ligand from string like 'NH3 (x6)'"""
    ligand_str = ligand_str.upper().replace('(', '').replace(')', '').replace('X', '')
    
    for lig in LIGAND_STRENGTH.keys():
        if lig.upper() in ligand_str:
            return lig
    
    return 'H2O'  # default

def predict_lambda_max_cft(metal, ox_state, geometry, ligands_str):
    """
    Predict lambda_max using simplified CFT
    
    Returns: predicted lambda_max in nm
    """
    ligand = extract_ligand_type(ligands_str)
    ligand_strength = LIGAND_STRENGTH.get(ligand, 1.0)

    geom_factor = GEOMETRY_FACTOR.get(geometry.lower(), 1.0)
    
    delta = ligand_strength * geom_factor * (ox_state * 1000)  
    
    if geometry.lower() == 'tetrahedral':
        constant = 400000 
    else:
        constant = 600000 
    
    lambda_max = constant / delta
    lambda_max = max(200, min(lambda_max, 1200))
    
    return round(lambda_max, 1)

def predict_color_from_lambda(lambda_nm):
    """Convert wavelength to observed color (complementary)"""
    if lambda_nm < 380:
        return 'colorless (UV)'
    elif lambda_nm < 450:
        return 'yellow'
    elif lambda_nm < 495:
        return 'orange'
    elif lambda_nm < 570:
        return 'red'
    elif lambda_nm < 590:
        return 'purple'
    elif lambda_nm < 620:
        return 'blue'
    elif lambda_nm < 750:
        return 'green'
    else:
        return 'colorless (IR)'


def test_baseline():
    """Test baseline model on known complexes"""
    test_cases = [
        ('[Co(NH3)6]3+', 'Co', 3, 'octahedral', 'NH3 (x6)', 475, 'yellow-orange'),
        ('[Ni(H2O)6]2+', 'Ni', 2, 'octahedral', 'H2O (x6)', 720, 'green'),
        ('[Cu(H2O)6]2+', 'Cu', 2, 'octahedral', 'H2O (x6)', 810, 'blue'),
        ('[CoCl4]2-', 'Co', 2, 'tetrahedral', 'Cl- (x4)', 690, 'blue')
    ]
    
    print("=" * 60)
    print("BASELINE MODEL TEST (Crystal Field Theory)")
    print("=" * 60)
    print()
    
    total_error = 0
    
    for name, metal, ox, geom, ligs, real_lambda, real_color in test_cases:
        pred_lambda = predict_lambda_max_cft(metal, ox, geom, ligs)
        pred_color = predict_color_from_lambda(pred_lambda)
        
        error = abs(pred_lambda - real_lambda)
        total_error += error
        
        print(f"Complex: {name}")
        print(f"  Predicted: λmax = {pred_lambda:.0f} nm ({pred_color})")
        print(f"  Actual:    λmax = {real_lambda:.0f} nm ({real_color})")
        print(f"  Error:     {error:.0f} nm")
        print()
    
    mae = total_error / len(test_cases)
    print("=" * 60)
    print(f"Mean Absolute Error (MAE): {mae:.1f} nm")
    print("=" * 60)


if __name__ == '__main__':
    test_baseline()