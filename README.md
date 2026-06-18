# Transition Metal Complex Color Predictor

🧪 ML-based tool for predicting absorption spectra (λmax) and observed color of transition metal coordination compounds.

## Project Status
🚧 **In Development** (Started: June 2024)

### Current Progress
- [x] Project structure
- [x] Initial dataset collection
  - [x] Co complexes (4/50)
  - [x] Ni complexes (2/50)
  - [x] Cu complexes (2/40)
  - [x] Fe complexes (1/30)
  - [x] Other metals (1/30) - Cr, Mn
- [x] Baseline CFT model ✅
- [ ] ML model training (next step)
- [ ] Web application
- [ ] Dataset publication (Zenodo)

## What This Project Does

**Problem:** Predicting the color of transition metal coordination compounds is difficult:
- Crystal Field Theory (CFT) gives rough estimates
- Quantum chemistry (DFT) is accurate but computationally expensive

**Solution:** Train a machine learning model on experimental UV-Vis data to predict:
- λmax (wavelength of maximum absorption)
- Observed color of the complex

## Dataset Schema
Each complex in the dataset includes:
- Metal center (element, oxidation state)
- Ligands (type, count, denticity)
- Coordination geometry (octahedral, tetrahedral, square planar, etc.)
- Experimental λmax (nm) from UV-Vis spectroscopy
- Observed color
- Solvent used
- Literature reference (DOI)

## Current Dataset Stats
- **Total complexes:** 10
- **Metals covered:** Co, Ni, Cu, Fe, Cr, Mn
- **Geometries:** Octahedral (8), Tetrahedral (2)
- **Baseline Model MAE:** ~XX nm (to be calculated)

## Tech Stack
- **Data collection & processing:** pandas, numpy
- **Machine Learning:** scikit-learn (Random Forest, Gradient Boosting)
- **Visualization:** matplotlib, seaborn
- **Web API:** FastAPI
- **Language:** Python 3.10+

## Installation
```bash
git clone https://github.com/rokuromizu34/transition-metal-predictor.git
cd transition-metal-predictor
python -m pip install -r requirements.txt
