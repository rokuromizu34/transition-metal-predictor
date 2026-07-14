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
| Dataset size | ~30 elements | 15,000+ materials |
| Features | ~5 | 130+ (MAGPIE descriptors) |
| Model type | In development | XGBoost |
| R² score | TBD | ~0.94 |
| MAE | TBD | ~0.21 eV |

---

##  Roadmap

- [ ] Connect Materials Project API (10,000+ materials)
- [ ] Advanced feature engineering (MAGPIE descriptors)
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
| Manual collection | ~30 elements | ✅ Current |
| Materials Project API | 150,000+ | 🔄 Planned |
| AFLOW database | 3,500,000+ | 🔄 Planned |

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

## 🙏 Acknowledgements

- [Materials Project](https://materialsproject.org/)
- [Streamlit](https://streamlit.io/)
- [matminer](https://hackingmaterials.lbl.gov/matminer/)

---

If you find this project useful — give it a star on GitHub!
It helps others discover the project.