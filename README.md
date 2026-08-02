# ⚗️ Transition Metal Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

> Machine learning tool for predicting physicochemical properties
> of transition metals and their compounds

## 🌐 Live Demo
👉 [transition-metal-predictor.streamlit.app](https://transition-metal-predictor-egpj469g6to22fqsv3yd9w.streamlit.app/)

---

## 📌 About The Project

Transition Metal Predictor is a machine learning web application
that predicts physicochemical properties of transition metals
based on their characteristics.

Transition metals are a fascinating group of elements that form
the backbone of modern materials science — from semiconductors
to catalysts and structural alloys. This tool aims to accelerate
materials discovery by providing instant ML-based predictions.

### What can it do?
- 🔮 Predict properties of transition metals and their compounds
- 📊 Visualize prediction results interactively
- ⚡ Run directly in browser — no installation needed
- 🔍 Explain why the model made a specific prediction

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| Streamlit | Web interface |
| Scikit-learn | ML models |
| Pandas | Data processing |
| Plotly | Interactive visualization |
| XGBoost | Gradient boosting model |

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

# 1. Clone the repository
git clone https://github.com/rokuromizu34/transition-metal-predictor
cd transition-metal-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py

App will open at http://localhost:8501

---

##  Model Status

> ⚠️ Project is under active development

| Parameter | Current | Target |
|-----------|---------|--------|
| Dataset size | 93 complexes | 150-200 complexes |
| Features | 10 | 15+ (incl. quantum-chemical descriptors) |
| Model type | ExtraTreesRegressor | XGBoost / ensemble comparison |
| KFold MAE | 68.4 nm | < 50 nm |
| Leave-one-metal-out MAE | 116.0 nm (honest metric) | < 90 nm |

---

## Validation

📄 **[Full Research Report (PDF)](docs/Transition_Metal_Predictor_Report.pdf)** — hypothesis, methods, results, limitations, and future work.

The model is evaluated two ways, since a single accuracy number can be
misleading depending on how the train/test split is built:

| Validation method | Splits by | MAE (nm) | What it actually measures |
|---|---|---|---|
| KFold (`models/retrain.py`) | random rows | **68.4** | Interpolation between complexes of metals the model has already seen elsewhere in training |
| GroupKFold by source (`models/evaluate_groupcv.py`) | literature source | — | Generalization across sources, still with every metal present in both train and test |
| **Leave-one-metal-out (`models/evaluate_metal_holdout.py`)** | metal | **116.0** | Generalization to a **transition metal the model has never seen at all** |

The leave-one-metal-out MAE is ~1.7x higher than the standard KFold
MAE. This is the honest number: standard KFold shuffles all complexes
together, so nearly every fold contains examples of nearly every
metal, meaning the model only has to interpolate within a metal it
already knows. Leave-one-metal-out instead trains on 7 metals and
predicts the 8th cold — the same situation the app faces with a truly
new metal in production.

Per-metal breakdown (see `data/processed/metal_holdout_results.csv`
and `data/processed/metal_holdout_plot.png`):

| Metal | Test samples | MAE (nm) |
|---|---|---|
| Cr | 15 | 33.4 |
| V | 2 | 44.1 |
| Ti | 3 | 68.7 |
| Co | 21 | 91.7 |
| Ni | 16 | 108.5 |
| Fe | 13 | 126.1 |
| Cu | 15 | 147.1 |
| Mn | 8 | 310.7 |

Mn stands out as the hardest metal to generalize to — likely because
its d5 electron configuration at Mn²⁺ has few close analogues
elsewhere in the current 93-complex dataset. This is a concrete
priority for the next round of data collection.

---

##  Roadmap

- [ ] Expand dataset via manual literature curation (target: 150-200 complexes, prioritizing underrepresented metals Mn/V/Ti)
- [ ] Advanced feature engineering (quantum-chemical descriptors, e.g. HOMO-LUMO gap via a lightweight method such as xtb)
- [ ] XGBoost model with cross-validation
- [ ] SHAP explainability for predictions
- [ ] Periodic table visualization
- [ ] Predict band gap, formation energy, bulk modulus
- [ ] Export results to CSV/JSON
- [ ] REST API endpoint

---

## Contributing & Collaboration

This project is open for collaboration!

### We are especially looking for help with:
- 🧪 **Chemistry / Materials Science** — data validation and domain expertise
- 🤖 **Machine Learning** — model improvement and feature engineering
- 🎨 **UI/UX** — interface improvements

### How to contribute:
1. Fork the repository
2. Create your branch
   git checkout -b feature/your-feature
3. Commit your changes
   git commit -m 'Add some feature'
4. Push to the branch
   git push origin feature/your-feature
5. Open a Pull Request

### Looking for academic collaborators
If you are a researcher in computational chemistry,
materials science or related fields and interested
in collaboration — please reach out!

We are particularly interested in:
- Validating model predictions against experimental data
- Expanding the dataset with DFT calculations
- Co-authoring a research paper on the results

---

##  Data Sources

| Source | Materials | Status |
|--------|-----------|--------|
| Miessler & Tarr, *Inorganic Chemistry* (2014) | 93 complexes | ✅ Current |
| Housecroft & Sharpe / Shriver & Atkins textbook tables | ~50-100 additional complexes | 🔄 Planned |

> **Note:** Materials Project and AFLOW are DFT databases of solid-state
> inorganic materials (band gaps, formation energies) and do not contain
> molecular-complex UV-Vis spectral data (λmax), so they are not usable
> sources for expanding this dataset. Expansion requires manual curation
> from coordination-chemistry literature instead.

---

## References

- [Materials Project](https://materialsproject.org/) — 
  High-throughput computational materials science
- [MAGPIE Descriptors](https://hackingmaterials.lbl.gov/matminer/) — 
  Materials-Agnostic Platform for Informatics and Exploration
- [matminer](https://hackingmaterials.lbl.gov/matminer/) — 
  Python library for data mining in materials science

---

## Contact

- **GitHub:** [@rokuromizu34](https://github.com/rokuromizu34)
- **Issues:** [Report a bug](https://github.com/rokuromizu34/transition-metal-predictor/issues)
- **Discussions:** [Ask a question](https://github.com/rokuromizu34/transition-metal-predictor/discussions)

---

## 📄 License

Distributed under the MIT License.
See `LICENSE` for more information.

---
## 📊 Data Quality

The dataset underwent a cleaning process to remove duplicate 
and conflicting entries:

- **Original entries:** 107
- **Duplicates removed:** 14 (exact duplicates + conflicting 
  values resolved by keeping original literature values)
- **Final clean dataset:** 93 unique transition metal complexes

This ensures no data leakage during cross-validation and 
consistent, non-contradictory training signal for the model.

## 🙏 Acknowledgements

- [Materials Project](https://materialsproject.org/)
- [Streamlit](https://streamlit.io/)
- [matminer](https://hackingmaterials.lbl.gov/matminer/)

---

If you find this project useful — give it a star on GitHub!
It helps others discover the project.
