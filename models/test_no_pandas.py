"""
Test baseline model on data without pandas
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from baseline import predict_lambda_max_cft, predict_color_from_lambda


# Данные встроены в код (вместо CSV)
COMPLEXES_DATA = [
    {
        'complex_name': '[Co(NH3)6]Cl3',
        'metal': 'Co',
        'ox_state': 3,
        'ligands': 'NH3 (x6)',
        'geometry': 'octahedral',
        'lambda_max': 475,
        'color': 'yellow-orange',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Ni(H2O)6]Cl2',
        'metal': 'Ni',
        'ox_state': 2,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 720,
        'color': 'green',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Cu(H2O)6]SO4',
        'metal': 'Cu',
        'ox_state': 2,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 810,
        'color': 'blue',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Co(H2O)6]Cl2',
        'metal': 'Co',
        'ox_state': 2,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 510,
        'color': 'pink',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Cr(H2O)6]Cl3',
        'metal': 'Cr',
        'ox_state': 3,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 575,
        'color': 'violet',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[CoCl4]2-',
        'metal': 'Co',
        'ox_state': 2,
        'ligands': 'Cl- (x4)',
        'geometry': 'tetrahedral',
        'lambda_max': 690,
        'color': 'blue',
        'solvent': 'EtOH'
    },
    {
        'complex_name': '[CuCl4]2-',
        'metal': 'Cu',
        'ox_state': 2,
        'ligands': 'Cl- (x4)',
        'geometry': 'tetrahedral',
        'lambda_max': 430,
        'color': 'yellow',
        'solvent': 'EtOH'
    },
    {
        'complex_name': '[Ni(NH3)6]Cl2',
        'metal': 'Ni',
        'ox_state': 2,
        'ligands': 'NH3 (x6)',
        'geometry': 'octahedral',
        'lambda_max': 570,
        'color': 'blue-violet',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Fe(H2O)6]SO4',
        'metal': 'Fe',
        'ox_state': 2,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 305,
        'color': 'pale-green',
        'solvent': 'H2O'
    },
    {
        'complex_name': '[Mn(H2O)6]Cl2',
        'metal': 'Mn',
        'ox_state': 2,
        'ligands': 'H2O (x6)',
        'geometry': 'octahedral',
        'lambda_max': 500,
        'color': 'pale-pink',
        'solvent': 'H2O'
    }
]


def test_baseline_on_data():
    """Test baseline CFT model on our dataset"""
    
    print("=" * 70)
    print("TESTING BASELINE MODEL ON 10 COMPLEXES")
    print("=" * 70)
    print(f"Total complexes: {len(COMPLEXES_DATA)}")
    print("=" * 70)
    print()
    
    errors = []
    
    for idx, data in enumerate(COMPLEXES_DATA):
        # Get data
        name = data['complex_name']
        metal = data['metal']
        ox_state = data['ox_state']
        geometry = data['geometry']
        ligands = data['ligands']
        real_lambda = data['lambda_max']
        real_color = data['color']
        
        # Predict
        pred_lambda = predict_lambda_max_cft(metal, ox_state, geometry, ligands)
        pred_color = predict_color_from_lambda(pred_lambda)
        
        # Calculate error
        error = abs(pred_lambda - real_lambda)
        errors.append(error)
        
        # Print results
        print(f"{idx+1}. {name}")
        print(f"   Metal: {metal}({ox_state:+d}), Geometry: {geometry}")
        print(f"   Predicted: λmax = {pred_lambda:.0f} nm ({pred_color})")
        print(f"   Actual:    λmax = {real_lambda:.0f} nm ({real_color})")
        print(f"   Error:     {error:.0f} nm ({error/real_lambda*100:.1f}%)")
        print()
    
    # Summary statistics
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    mean_lambda = sum(data['lambda_max'] for data in COMPLEXES_DATA) / len(COMPLEXES_DATA)
    
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total complexes tested: {len(COMPLEXES_DATA)}")
    print(f"Mean Absolute Error (MAE): {mean_error:.1f} nm")
    print(f"Max error: {max_error:.0f} nm")
    print(f"Min error: {min_error:.0f} nm")
    print(f"Mean relative error: {(mean_error / mean_lambda * 100):.1f}%")
    
    # Analysis by geometry
    oct_errors = [errors[i] for i, d in enumerate(COMPLEXES_DATA) if d['geometry'] == 'octahedral']
    tet_errors = [errors[i] for i, d in enumerate(COMPLEXES_DATA) if d['geometry'] == 'tetrahedral']
    
    print()
    print("BY GEOMETRY:")
    print(f"  Octahedral complexes (n={len(oct_errors)}): MAE = {sum(oct_errors)/len(oct_errors):.1f} nm")
    if tet_errors:
        print(f"  Tetrahedral complexes (n={len(tet_errors)}): MAE = {sum(tet_errors)/len(tet_errors):.1f} nm")
    
    print("=" * 70)
    
    return mean_error


if __name__ == '__main__':
    mae = test_baseline_on_data()