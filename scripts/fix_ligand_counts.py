import pandas as pd

path = "data/raw/complexes_raw.csv"
df = pd.read_csv(path)

df.loc[df["complex_name"] == "[Co(NH3)5Cl]2+", "ligands"] = "NH3*5+Cl*1"
df.loc[df["complex_name"] == "[Co(NH3)4Cl2]+", "ligands"] = "NH3*4+Cl*2"

df.to_csv(path, index=False)
print("Saved:", path)
