"""
Improved baseline model with better calibration (v2)
"""

# Улучшенный спектрохимический ряд
LIGAND_STRENGTH = {
    'I-': 0.6,
    'Br-': 0.7,
    'Cl-': 0.78,
    'F-': 0.9,
    'OH-': 0.95,
    'H2O': 1.0,    # базовая единица
    'NH3': 1.25,
    'en': 1.4,     # ethylenediamine
    'bpy': 1.5,    # bipyridine
    'phen': 1.5,   # phenanthroline
    'NO2-': 1.6,
    'CN-': 1.7,
    'CO': 1.8
}

METAL_FACTORS = {
    'Ti': 0.8, 'V': 0.85, 'Cr': 0.9, 'Mn': 1.1, 'Fe': 1.2,
    'Co': 1.0, 'Ni': 0.8, 'Cu': 0.7, 'Zn': 1.0
}

OXIDATION_FACTORS = {
    1: 1.3, 2: 1.0, 3: 0.85, 4: 0.7, 5: 0.6, 6: 0.5
}


def extract_ligand_type(ligand_str):
    """Extract dominant ligand type"""
    ligand_str = ligand_str.upper().replace('(', '').replace(')', '').replace('X', '')
    
    # Приоритет по силе поля
    if 'CN' in ligand_str:
        return 'CN-'
    elif 'CO' in ligand_str:
        return 'CO'
    elif 'NH3' in ligand_str:
        return 'NH3'
    elif 'EN' in ligand_str:
        return 'en'
    elif 'H2O' in ligand_str:
        return 'H2O'
    elif 'CL' in ligand_str:
        return 'Cl-'
    elif 'BR' in ligand_str:
        return 'Br-'
    elif 'I' in ligand_str:
        return 'I-'
    elif 'F' in ligand_str:
        return 'F-'
    else:
        return 'H2O'  # default


def predict_lambda_max_v2(metal, ox_state, geometry, ligands_str):
    """
    Improved CFT prediction with better calibration
    
    Returns: predicted lambda_max in nm
    """
    # Base parameters
    ligand = extract_ligand_type(ligands_str)
    ligand_strength = LIGAND_STRENGTH.get(ligand, 1.0)
    metal_factor = METAL_FACTORS.get(metal, 1.0)
    ox_factor = OXIDATION_FACTORS.get(ox_state, 1.0)
    
    # Geometry-specific calculations
    if geometry.lower() == 'octahedral':
        base_lambda = 600
        lambda_max = base_lambda / (ligand_strength * ox_factor) * metal_factor
        
    elif geometry.lower() == 'tetrahedral':
        # Тетраэдрические: базовая λ = 500 нм для [MCl4]2-
        base_lambda = 500
        lambda_max = base_lambda / (ligand_strength * ox_factor * 0.44) * metal_factor
        
    else:
        base_lambda = 550
        lambda_max = base_lambda / ligand_strength * metal_factor
    
    lambda_max = max(250, min(lambda_max, 1000))
    
    return round(lambda_max, 1)


def predict_color_from_lambda(lambda_nm):
    """Convert wavelength to observed color"""
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


def test_baseline_v2():
    """Test improved baseline model"""
    
    # Test cases
    test_cases = [
        ('[Co(NH3)6]3+', 'Co', 3, 'octahedral', 'NH3 (x6)', 475, 'yellow-orange'),
        ('[Ni(H2O)6]2+', 'Ni', 2, 'octahedral', 'H2O (x6)', 720, 'green'),
        ('[Cu(H2O)6]2+', 'Cu', 2, 'octahedral', 'H2O (x6)', 810, 'blue'),
        ('[Co(H2O)6]2+', 'Co', 2, 'octahedral', 'H2O (x6)', 510, 'pink'),
        ('[Cr(H2O)6]3+', 'Cr', 3, 'octahedral', 'H2O (x6)', 575, 'violet'),
        ('[CoCl4]2-', 'Co', 2, 'tetrahedral', 'Cl- (x4)', 690, 'blue'),
        ('[CuCl4]2-', 'Cu', 2, 'tetrahedral', 'Cl- (x4)', 430, 'yellow'),
        ('[Ni(NH3)6]2+', 'Ni', 2, 'octahedral', 'NH3 (x6)', 570, 'blue-violet'),
        ('[Fe(H2O)6]2+', 'Fe', 2, 'octahedral', 'H2O (x6)', 305, 'pale-green'),
        ('[Mn(H2O)6]2+', 'Mn', 2, 'octahedral', 'H2O (x6)', 500, 'pale-pink'),
    ]
    
    print("=" * 70)
    print("IMPROVED BASELINE MODEL TEST (v2)")
    print("=" * 70)
    print()
    
    errors = []
    
    for name, metal, ox, geom, ligs, real_lambda, real_color in test_cases:
        pred_lambda = predict_lambda_max_v2(metal, ox, geom, ligs)
        pred_color = predict_color_from_lambda(pred_lambda)
        
        error = abs(pred_lambda - real_lambda)
        errors.append(error)
        
        print(f"{name}")
        print(f"  Metal: {metal}({ox:+d}), Geometry: {geom}")
        print(f"  Predicted: λmax = {pred_lambda:.0f} nm ({pred_color})")
        print(f"  Actual:    λmax = {real_lambda:.0f} nm ({real_color})")
        print(f"  Error:     {error:.0f} nm ({error/real_lambda*100:.1f}%)")
        print()
    
    # Statistics
    mae = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    
    #  geometry
    oct_errors = [errors[i] for i, (_, _, _, geom, _, _, _) in enumerate(test_cases) if geom == 'octahedral']
    tet_errors = [errors[i] for i, (_, _, _, geom, _, _, _) in enumerate(test_cases) if geom == 'tetrahedral']
    
    print("=" * 70)
    print("SUMMARY STATISTICS (v2)")
    print("=" * 70)
    print(f"Total complexes: {len(test_cases)}")
    print(f"Mean Absolute Error: {mae:.1f} nm")
    print(f"Max error: {max_error:.0f} nm")
    print(f"Min error: {min_error:.0f} nm")
    print()
    print("BY GEOMETRY:")
    if oct_errors:
        print(f"  Octahedral (n={len(oct_errors)}): MAE = {sum(oct_errors)/len(oct_errors):.1f} nm")
    if tet_errors:
        print(f"  Tetrahedral (n={len(tet_errors)}): MAE = {sum(tet_errors)/len(tet_errors):.1f} nm")
    print("=" * 70)
    
    return mae


if __name__ == '__main__':
    mae = test_baseline_v2()
    print(f"\n IMPROVED MODEL MAE: {mae:.1f} nm")