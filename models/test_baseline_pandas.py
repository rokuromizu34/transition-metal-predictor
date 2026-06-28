"""
Test baseline model on collected dataset using pandas
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from baseline import predict_lambda_max_cft, predict_color_from_lambda


def test_baseline_on_dataset():
    """Test baseline CFT model on collected data"""
    
    # Load dataset
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'complexes_raw.csv'
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} complexes from CSV")
    except FileNotFoundError:
        print(f"CSV file not found at {data_path}")
        print("Using hardcoded data instead...")
        
        # Fallback to hardcoded data
        data = {
            'complex_name': ['[Co(NH3)6]Cl3', '[Ni(H2O)6]Cl2', '[Cu(H2O)6]SO4', 
                           '[Co(H2O)6]Cl2', '[Cr(H2O)6]Cl3', '[CoCl4]2-',
                           '[CuCl4]2-', '[Ni(NH3)6]Cl2', '[Fe(H2O)6]SO4', '[Mn(H2O)6]Cl2'],
            'metal': ['Co', 'Ni', 'Cu', 'Co', 'Cr', 'Co', 'Cu', 'Ni', 'Fe', 'Mn'],
            'ox_state': [3, 2, 2, 2, 3, 2, 2, 2, 2, 2],
            'ligands': ['NH3 (x6)', 'H2O (x6)', 'H2O (x6)', 'H2O (x6)', 'H2O (x6)',
                       'Cl- (x4)', 'Cl- (x4)', 'NH3 (x6)', 'H2O (x6)', 'H2O (x6)'],
            'geometry': ['octahedral', 'octahedral', 'octahedral', 'octahedral', 'octahedral',
                        'tetrahedral', 'tetrahedral', 'octahedral', 'octahedral', 'octahedral'],
            'lambda_max': [475, 720, 810, 510, 575, 690, 430, 570, 305, 500],
            'color': ['yellow-orange', 'green', 'blue', 'pink', 'violet', 
                     'blue', 'yellow', 'blue-violet', 'pale-green', 'pale-pink']
        }
        df = pd.DataFrame(data)
    
    print("=" * 70)
    print("TESTING BASELINE MODEL ON COLLECTED DATA")
    print("=" * 70)
    print(f"Total complexes: {len(df)}")
    print("=" * 70)
    print()
    
    # Add prediction columns
    df['pred_lambda'] = df.apply(
        lambda row: predict_lambda_max_cft(
            row['metal'], row['ox_state'], 
            row['geometry'], row['ligands']
        ), axis=1
    )
    
    df['pred_color'] = df['pred_lambda'].apply(predict_color_from_lambda)
    df['error'] = abs(df['pred_lambda'] - df['lambda_max'])
    df['rel_error'] = (df['error'] / df['lambda_max']) * 100
    
    # Print results
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['complex_name']}")
        print(f"   Metal: {row['metal']}({row['ox_state']:+d}), Geometry: {row['geometry']}")
        print(f"   Predicted: λmax = {row['pred_lambda']:.0f} nm ({row['pred_color']})")
        print(f"   Actual:    λmax = {row['lambda_max']:.0f} nm ({row['color']})")
        print(f"   Error:     {row['error']:.0f} nm ({row['rel_error']:.1f}%)")
        print()
    
    # Summary statistics
    mae = df['error'].mean()
    max_error = df['error'].max()
    min_error = df['error'].min()
    mean_rel_error = df['rel_error'].mean()
    
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total complexes tested: {len(df)}")
    print(f"Mean Absolute Error (MAE): {mae:.1f} nm")
    print(f"Max error: {max_error:.0f} nm")
    print(f"Min error: {min_error:.0f} nm")
    print(f"Mean relative error: {mean_rel_error:.1f}%")
    
    # Analysis by geometry
    oct_df = df[df['geometry'] == 'octahedral']
    tet_df = df[df['geometry'] == 'tetrahedral']
    
    print()
    print("BY GEOMETRY:")
    print(f"  Octahedral complexes (n={len(oct_df)}): MAE = {oct_df['error'].mean():.1f} nm")
    if len(tet_df) > 0:
        print(f"  Tetrahedral complexes (n={len(tet_df)}): MAE = {tet_df['error'].mean():.1f} nm")
    
    # Analysis by metal
    print()
    print("BY METAL:")
    for metal in df['metal'].unique():
        metal_df = df[df['metal'] == metal]
        print(f"  {metal} complexes (n={len(metal_df)}): MAE = {metal_df['error'].mean():.1f} nm")
    
    print("=" * 70)
    
    return mae, df


if __name__ == '__main__':
    mae, results_df = test_baseline_on_dataset()
    print(f"\n Test completed! Final MAE: {mae:.1f} nm")