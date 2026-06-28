# ТЫ ПРАВА! Я убрал чтобы не было ошибки с двойным текстом.

## Давай вернём подробное README!

---

### Действие:

1. **Ctrl+A** в README.md
2. **Delete**
3. Скопируй текст ниже **АККУРАТНО**
4. **Ctrl+V**
5. **Ctrl+S**

### КОПИРУЙ:

```
# Transition Metal Color Predictor

ML model that predicts absorption wavelength (lambda max) and color of transition metal complexes.

**ML outperforms Crystal Field Theory by 64%.**

## Results

| Model | MAE (nm) | Description |
|-------|----------|-------------|
| Crystal Field Theory (baseline) | 110.6 | Rule-based spectrochemical series |
| Random Forest (ML) | 40.1 | Trained on 91 complexes |

## Dataset

First open-source machine-readable dataset of experimental lambda max values for coordination compounds.

- 91 complexes across 8 metals (Ti, V, Cr, Mn, Fe, Co, Ni, Cu)
- 3 geometries: octahedral, tetrahedral, square planar
- 20 ligand types: from I (weak field) to CN (strong field)
- Source: Miessler and Tarr, Inorganic Chemistry, 5th ed. (2014)

## Key Findings

1. ML improves predictions by 64% compared to Crystal Field Theory
2. Counter-ions do not affect lambda max (confirmed across dataset)
3. Most important features: ligand strength (27%), d-electron count (27%)
4. Best predictions: Cr complexes (MAE = 26 nm), Mn (MAE = 29 nm)
5. Hardest to predict: Fe complexes (MAE = 91 nm) due to spin-state effects

## How It Works

Input: Metal + Ligand + Geometry
  -> Feature Engineering (10 numerical features)
  -> Random Forest Model (200 trees)
  -> Output: lambda max (nm) and predicted Color

## Features

| Feature | Importance | Description |
|---------|-----------|-------------|
| ligand_strength | 27.0% | Spectrochemical series value |
| d_electrons | 26.7% | Number of d-electrons |
| effective_field | 18.2% | Ligand strength x geometry factor |
| metal_Z | 14.4% | Atomic number of metal |
| ox_state | 9.4% | Oxidation state |

## MAE by Metal

| Metal | N | Baseline | ML | Improvement |
|-------|---|----------|-----|-------------|
| Co | 21 | 64 nm | 32 nm | 2.0x |
| Ni | 16 | 81 nm | 33 nm | 2.5x |
| Cu | 15 | 171 nm | 43 nm | 4.0x |
| Fe | 11 | 261 nm | 91 nm | 2.9x |
| Cr | 15 | 66 nm | 26 nm | 2.6x |
| Mn | 8 | 89 nm | 29 nm | 3.1x |

## Quick Start

git clone https://github.com/rokuromizu34/transition-metal-predictor.git
cd transition-metal-predictor
py models/feature_engineering.py
py models/train_ml_models.py
py models/honest_comparison.py

## Project Structure

transition-metal-predictor/
  data/
    raw/complexes_raw.csv          (91 complexes)
    processed/                     (ML-ready features)
  models/
    baseline_v3.py                 (CFT baseline)
    feature_engineering.py         (Feature extraction)
    train_ml_models.py             (ML training)
    honest_comparison.py           (Fair comparison)
    best_ml_model.pkl              (Trained model)
  app/                             (Web application - coming)

## Tech Stack

- Data: pandas, numpy
- ML: scikit-learn (Random Forest, Gradient Boosting)
- Web: Streamlit (planned)
- Language: Python 3.10+

## Author

Research project by Olga — exploring computational chemistry and machine learning.

## License

MIT License
```

