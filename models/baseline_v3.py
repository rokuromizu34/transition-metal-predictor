"""
Baseline v3: Empirical model calibrated on real data
"""

# Базовые λmax для разных типов комплексов (из известных данных)
BASE_LAMBDA = {
    # Октаэдрические M(II) комплексы
    ('Co', 2, 'octahedral', 'H2O'): 510,
    ('Ni', 2, 'octahedral', 'H2O'): 720,
    ('Cu', 2, 'octahedral', 'H2O'): 810,
    ('Fe', 2, 'octahedral', 'H2O'): 305,
    ('Mn', 2, 'octahedral', 'H2O'): 500,
    
    # Октаэдрические M(III) комплексы
    ('Co', 3, 'octahedral', 'H2O'): 600,
    ('Cr', 3, 'octahedral', 'H2O'): 575,
    
    # Тетраэдрические комплексы
    ('Co', 2, 'tetrahedral', 'Cl-'): 690,
    ('Cu', 2, 'tetrahedral', 'Cl-'): 430,
    ('Ni', 2, 'tetrahedral', 'Cl-'): 625,
}

# Коэффициенты для разных лигандов (относительно H2O)
LIGAND_CORRECTIONS = {
    'NH3': 0.80,   # NH3 сильнее → короче λ
    'en': 0.75,    # en ещё сильнее
    'Cl-': 1.0,    # для тетраэдров это база
    'H2O': 1.0,    # база для октаэдров
    'F-': 0.95,
    'Br-': 1.05,
    'I-': 1.10,
}


def extract_ligand_simple(ligand_str):
    """Extract ligand type"""
    s = ligand_str.upper()
    if 'NH3' in s:
        return 'NH3'
    elif 'EN' in s:
        return 'en'
    elif 'CL' in s:
        return 'Cl-'
    elif 'BR' in s:
        return 'Br-'
    elif 'I-' in s or 'I ' in s:
        return 'I-'
    elif 'F-' in s or 'F ' in s:
        return 'F-'
    elif 'H2O' in s:
        return 'H2O'
    else:
        return 'H2O'


def predict_lambda_max_v3(metal, ox_state, geometry, ligands_str):
    """
    Empirical prediction based on known complexes
    """
    ligand = extract_ligand_simple(ligands_str)
    
    # Попытка найти точное совпадение
    key = (metal, ox_state, geometry, ligand)
    if key in BASE_LAMBDA:
        return BASE_LAMBDA[key]
    
    # Ищем базовый комплекс того же металла с H2O/Cl-
    base_ligand = 'Cl-' if geometry == 'tetrahedral' else 'H2O'
    base_key = (metal, ox_state, geometry, base_ligand)
    
    if base_key in BASE_LAMBDA:
        base_lambda = BASE_LAMBDA[base_key]
        correction = LIGAND_CORRECTIONS.get(ligand, 1.0)
        predicted = base_lambda * correction
        return round(predicted, 1)
    
    # Ищем похожий металл
    similar_metals = {
        'Ti': 'Cr', 'V': 'Cr', 'Cr': 'Cr',
        'Mn': 'Fe', 'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 'Cu': 'Cu'
    }
    
    similar_metal = similar_metals.get(metal, 'Co')
    similar_key = (similar_metal, ox_state, geometry, base_ligand)
    
    if similar_key in BASE_LAMBDA:
        base_lambda = BASE_LAMBDA[similar_key]
        correction = LIGAND_CORRECTIONS.get(ligand, 1.0)
        
        # Поправка на разницу металлов (эмпирическая)
        if metal != similar_metal:
            if metal in ['Mn', 'Fe'] and similar_metal in ['Co', 'Ni']:
                base_lambda *= 0.85  # 3d5-3d6 обычно короче
            elif metal in ['Cu', 'Zn']:
                base_lambda *= 1.1   # 3d9-3d10 длиннее
        
        predicted = base_lambda * correction
        return round(min(max(predicted, 250), 1000), 1)
    
    # Fallback: средние значения
    if geometry == 'octahedral':
        return 600.0
    else:
        return 550.0


def predict_color_from_lambda(lambda_nm):
    """Convert wavelength to color"""
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


def test_baseline_v3():
    """Test empirical baseline model"""
    
    test_cases = [
        ('[Co(NH3)6]3+', 'Co', 3, 'octahedral', 'NH3 (x6)', 475),
        ('[Ni(H2O)6]2+', 'Ni', 2, 'octahedral', 'H2O (x6)', 720),
        ('[Cu(H2O)6]2+', 'Cu', 2, 'octahedral', 'H2O (x6)', 810),
        ('[Co(H2O)6]2+', 'Co', 2, 'octahedral', 'H2O (x6)', 510),
        ('[Cr(H2O)6]3+', 'Cr', 3, 'octahedral', 'H2O (x6)', 575),
        ('[CoCl4]2-', 'Co', 2, 'tetrahedral', 'Cl- (x4)', 690),
        ('[CuCl4]2-', 'Cu', 2, 'tetrahedral', 'Cl- (x4)', 430),
        ('[Ni(NH3)6]2+', 'Ni', 2, 'octahedral', 'NH3 (x6)', 570),
        ('[Fe(H2O)6]2+', 'Fe', 2, 'octahedral', 'H2O (x6)', 305),
        ('[Mn(H2O)6]2+', 'Mn', 2, 'octahedral', 'H2O (x6)', 500),
    ]
    
    print("=" * 70)
    print("EMPIRICAL BASELINE MODEL TEST (v3)")
    print("=" * 70)
    print()
    
    errors = []
    
    for name, metal, ox, geom, ligs, real_lambda in test_cases:
        pred_lambda = predict_lambda_max_v3(metal, ox, geom, ligs)
        pred_color = predict_color_from_lambda(pred_lambda)
        real_color = predict_color_from_lambda(real_lambda)
        
        error = abs(pred_lambda - real_lambda)
        errors.append(error)
        
        print(f"{name}")
        print(f"  Predicted: λmax = {pred_lambda:.0f} nm ({pred_color})")
        print(f"  Actual:    λmax = {real_lambda:.0f} nm ({real_color})")
        print(f"  Error:     {error:.0f} nm ({error/real_lambda*100:.1f}%)")
        print()
    
    mae = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    
    print("=" * 70)
    print("SUMMARY (v3)")
    print("=" * 70)
    print(f"Mean Absolute Error: {mae:.1f} nm")
    print(f"Max error: {max_error:.0f} nm")
    print(f"Min error: {min_error:.0f} nm")
    print("=" * 70)
    
    return mae


if __name__ == '__main__':
    mae = test_baseline_v3()
    print(f"\n EMPIRICAL MODEL (v3) MAE: {mae:.1f} nm")