# Transition Metal Color Predictor

A web app that predicts the absorption maximum (λmax) and the perceived solution color of transition‑metal coordination complexes from simple chemistry inputs: **metal + oxidation state + ligands + geometry**.

**Live demo:** https://transition-metal-predictor-egpj469g6to22fqsv3yd9w.streamlit.app/

---

## What it does

**Input**
- Metal (Ti, V, Cr, Mn, Fe, Co, Ni, Cu)
- Oxidation state
- Ligand(s) (supports mixed ligands like `en+NH3`)
- Geometry (octahedral / tetrahedral / square planar)

**Output**
- Predicted **λmax (nm)**
- Absorbed color + perceived color swatches (HEX)
- Confidence hint (whether similar metal+oxidation was seen in training)

---

## Dataset

- **91** experimental complexes
- **8** metals: Ti, V, Cr, Mn, Fe, Co, Ni, Cu
- **3** geometries: octahedral, tetrahedral, square planar
- Ligand field strength encoded via a spectrochemical series scale (including mixed ligands)

**Source (literature):** Miessler & Tarr, *Inorganic Chemistry*, 5th ed. (2014).  
*(Dataset is compiled for educational/research purposes; please cite the source.)*

---

## Feature engineering (10 features)

Crystal‑Field‑Theory inspired descriptors:

- `metal_Z`
- `ox_state`
- `d_electrons`
- `ligand_strength` (supports stoichiometry like `NH3*5+Cl*1`)
- `coord_number`
- `geom_factor`
- `effective_field = ligand_strength × geom_factor`
- `is_octahedral`, `is_tetrahedral`, `is_square_planar`

---

## Model

- Model: **ExtraTreesRegressor** (scikit‑learn)
- Target transformation: model predicts **wavenumber (cm⁻¹)**, then converts back to **nm**

\[
\lambda(\text{nm}) = \frac{10^7}{\tilde{\nu}(\text{cm}^{-1})}
\]

---

## Performance (MAE in nm)

Because the dataset is small, performance depends on the evaluation protocol:

- **Random KFold(5):** MAE ≈ **64 nm**
- **GroupKFold (hold out entire metal):** MAE ≈ **101 nm**
- **GroupKFold (hold out metal + oxidation state):** MAE ≈ **112 nm**

**Known limitation:** Fe(III) (d⁵ high‑spin) systems often show UV / spin‑forbidden transitions that are not captured by the current feature set, so errors can be large.

---

## Quick start (local)

```bash
pip install -r requirements.txt
py models/feature_engineering.py
py models/train_fast.py
py -m streamlit run app/main.py