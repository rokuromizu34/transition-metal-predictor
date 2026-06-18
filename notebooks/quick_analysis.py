"""
Quick data analysis without Jupyter
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_dataset():
    """Quick analysis of the dataset"""
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'complexes_raw.csv'
    
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {len(df)} complexes")
    except FileNotFoundError:
        print("CSV file not found!")
        return
    
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    # Basic info
    print(f"Total complexes: {len(df)}")
    print(f"Metals: {df['metal'].value_counts().to_dict()}")
    print(f"Geometries: {df['geometry'].value_counts().to_dict()}")
    print(f"λmax range: {df['lambda_max'].min()}-{df['lambda_max'].max()} nm")
    print(f"Mean λmax: {df['lambda_max'].mean():.1f} nm")
    
    # Simple visualization (save plot)
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Lambda max by metal
    plt.subplot(2, 2, 1)
    df.boxplot(column='lambda_max', by='metal', ax=plt.gca())
    plt.title('λmax by Metal')
    plt.suptitle('')
    
    # Plot 2: Lambda max by geometry
    plt.subplot(2, 2, 2)
    df.boxplot(column='lambda_max', by='geometry', ax=plt.gca())
    plt.title('λmax by Geometry')
    plt.suptitle('')
    
    # Plot 3: Distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['lambda_max'], bins=10, edgecolor='black')
    plt.xlabel('λmax (nm)')
    plt.ylabel('Count')
    plt.title('λmax Distribution')
    
    # Plot 4: Metal vs λmax
    plt.subplot(2, 2, 4)
    for metal in df['metal'].unique():
        metal_data = df[df['metal'] == metal]
        plt.scatter(metal_data.index, metal_data['lambda_max'], 
                   label=metal, s=50)
    plt.xlabel('Complex Index')
    plt.ylabel('λmax (nm)')
    plt.title('λmax by Complex')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'dataset_overview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    analyze_dataset()