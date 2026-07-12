@'
# Transition Metal Color Predictor

A web app that predicts λmax and perceived solution color of transition-metal coordination complexes from: **metal + oxidation state + ligands + geometry**.

**Live demo:** https://transition-metal-predictor-egpj469g6to22fqsv3yd9w.streamlit.app/

---

## What it does

**Input:** Metal · Oxidation state · Ligand(s) · Geometry  
**Output:** λmax (nm) · Absorbed color · Perceived color (HEX) · Confidence hint

Supports mixed ligands (e.g. `en+NH3`).

---

## Dataset

- 91 experimental complexes
- 8 metals: Ti, V, Cr, Mn, Fe, Co, Ni, Cu
- 3 geometries: octahedral, tetrahedral, square planar
- 20 ligand types (weak-field I to strong-field CN)

Source: Miessler & Tarr, Inorganic Chemistry, 5th ed. (2014).

---

## Feature engineering (10 features)

metal_Z, ox_state, d_electrons, ligand_strength, coord_number, geom_factor, effective_field, is_octahedral, is_tetrahedral, is_square_planar

---

## Model

- ExtraTreesRegressor (scikit-learn)
- Trained on wavenumber (cm-1), converted back to nm: lambda = 1e7 / nu

---

## Performance

| Protocol | MAE (nm) |
|---|---|
| Random KFold(5) | 64 |
| GroupKFold by metal | 101 |
| GroupKFold by metal+ox | 112 |

Known limitation: Fe(III) d5 high-spin systems (UV/spin-forbidden transitions).

---

## Quick start

```bash
pip install -r requirements.txt
py models/feature_engineering.py
py models/train_fast.py
py -m streamlit run app/main.py