"""
Clean dataset: remove exact duplicates, 
flag conflicting entries for manual review
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "data/raw/complexes_raw.csv")

print(f"Original size: {len(df)} rows")

# Find exact duplicates (all columns same)
exact_dupes = df[df.duplicated(keep=False)]
print(f"\nExact duplicates found: {len(exact_dupes)}")
if len(exact_dupes) > 0:
    print(exact_dupes[['complex_name', 'lambda_max']])

# Find conflicting entries (same complex, different lambda_max)
key_cols = ['metal', 'ox_state', 'ligands', 'geometry']
grouped = df.groupby(key_cols)['lambda_max'].nunique()
conflicts = grouped[grouped > 1]

print(f"\n⚠️ CONFLICTING entries (same complex, different values):")
for idx in conflicts.index:
    mask = True
    for col, val in zip(key_cols, idx):
        mask = mask & (df[col] == val)
    print(df[mask][['complex_name', 'lambda_max', 'color', 'solvent']])
    print("---")

# Remove exact duplicates (keep first occurrence)
df_clean = df.drop_duplicates(
    subset=['metal', 'ox_state', 'ligands', 'geometry'],
    keep='first'
)

print(f"\nAfter removing duplicates: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} rows")

# Save cleaned version
df_clean.to_csv(
    ROOT / "data/raw/complexes_raw_clean.csv", 
    index=False
)
print(f"\n✅ Saved to complexes_raw_clean.csv")
print("⚠️ Review conflicts above BEFORE using this file!")