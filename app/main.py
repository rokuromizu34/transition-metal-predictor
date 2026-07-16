"""
Transition Metal Color Predictor - Main Streamlit App
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import streamlit as st

from constants import (
    LIGAND_STRENGTH, GEOMETRY_CN, GEOMETRY_FACTOR,
    METAL_Z, D_ELECTRONS, parse_ligand_strength
)
from color_utils import (
    KNOWN_COLORS, spectrum_to_hex,
    wavelength_to_absorbed_name
)

ROOT          = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "models" / "best_ml_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = list(joblib.load(FEATURES_PATH))
    return model, feature_names


@st.cache_resource
def load_meta():
    p = ROOT / "models" / "model_meta.pkl"
    return joblib.load(p) if p.exists() else {}


@st.cache_data
def load_raw_df():
    return pd.read_csv(ROOT / "data/raw/complexes_raw.csv")


def build_features(
    metal: str,
    ox: int,
    ligand_str: str,
    geometry: str
) -> pd.DataFrame:
    """
    Build feature vector for model prediction.
    Uses Crystal Field Theory descriptors.
    """
    lig_strength = parse_ligand_strength(ligand_str)
    geom_factor  = GEOMETRY_FACTOR.get(geometry, 1.0)

    features = {
        'metal_Z':          METAL_Z.get(metal, 26),
        'ox_state':         ox,
        'd_electrons':      D_ELECTRONS.get((metal, ox), 5),
        'ligand_strength':  lig_strength,
        'coord_number':     GEOMETRY_CN.get(geometry, 6),
        'geom_factor':      geom_factor,
        'effective_field':  lig_strength * geom_factor,
        'is_octahedral':    int(geometry == 'octahedral'),
        'is_tetrahedral':   int(geometry == 'tetrahedral'),
        'is_square_planar': int(geometry == 'square_planar'),
    }

    return pd.DataFrame([features]).reindex(columns=feature_names)


# ── Load everything ────────────────────────────────
model, feature_names = load_artifacts()
meta   = load_meta()
raw_df = load_raw_df()

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="Transition Metal Color Predictor",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Transition Metal Color Predictor")
st.caption(
    "Predicts the color of transition metal complexes "
    "using ML + Crystal Field Theory"
)

# ── Dataset statistics ───────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Complexes", len(raw_df))
with col2:
    st.metric("Metals", raw_df["metal"].nunique())
with col3:
    st.metric("Geometries", raw_df["geometry"].nunique())
with col4:
    st.metric("Model MAE", "~64 nm")

st.divider()

# ── Input widgets ─────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    metal = st.selectbox(
        "Metal",
        list(METAL_Z.keys()),
        index=list(METAL_Z.keys()).index("Co")
    )
    ox_state = st.number_input(
        "Oxidation state",
        min_value=1, max_value=9,
        value=3, step=1
    )

with c2:
    ligand = st.selectbox(
        "Ligand",
        list(LIGAND_STRENGTH.keys()),
        index=list(LIGAND_STRENGTH.keys()).index("NH3")
    )
    geometry = st.selectbox(
        "Geometry",
        list(GEOMETRY_CN.keys()),
        index=0
    )

custom = st.text_input(
    "Mixed ligands (e.g. NH3*4+en*2)",
    value=""
)
show_debug = st.checkbox("Show debug info", value=False)

# ── Prediction ──────────────────────────────────────
if st.button("PREDICT COLOR", type="primary",)
             