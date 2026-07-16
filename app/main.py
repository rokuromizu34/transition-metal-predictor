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


# ── Load everything ───────────────────────────────────────
model, feature_names = load_artifacts()
meta   = load_meta()
raw_df = load_raw_df()

# ── Page config ───────────────────────────────────────────
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
# ── Dataset statistics ────────────────────────────
st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Complexes",
        value=len(raw_df)        
    )
with col2:
    st.metric(
        label="Metals",
        value=raw_df["metal"].nunique() 
    )
with col3:
    st.metric(
        label="Geometries",
        value=raw_df["geometry"].nunique()  
    )
with col4:
    st.metric(
        label="Model MAE",
        value="~64 nm"
    )

st.divider()
# ── Prediction ────────────────────────────────────────────
if st.button("PREDICT COLOR", type="primary",
             use_container_width=True):

    lig_input = custom.strip() if custom.strip() else ligand
    X         = build_features(metal, int(ox_state),
                               lig_input, geometry)

    raw_pred = float(model.predict(X)[0])

    # Convert model output to wavelength in nm
    if meta.get("target") == "nm":
        lam = raw_pred
    elif meta.get("target") == "wavenumber_cm-1":
        lam = 1e7 / raw_pred
    else:
        # Auto-detect: wavenumbers are typically > 5000
        lam = raw_pred if raw_pred < 1000 else 1e7 / raw_pred

    # Confidence based on training data coverage
    seen = ((raw_df["metal"] == metal) &
            (raw_df["ox_state"] == int(ox_state))).any()
    confidence = (
        "Higher confidence — metal+oxidation state seen in training"
        if seen else
        "Lower confidence — metal+oxidation state not in training data"
    )

    # Color lookup or CIE calculation
    key = (metal, int(ox_state), lig_input, geometry)
    if key in KNOWN_COLORS:
        perc_hex, abs_hex, perc_name = KNOWN_COLORS[key]
        verified = True
    else:
        perc_hex, abs_hex, perc_name = spectrum_to_hex(
            lam, fwhm_nm=120.0
        )
        verified = False

    st.session_state["result"] = {
        "metal":     metal,
        "ox_state":  int(ox_state),
        "lig_input": lig_input,
        "geometry":  geometry,
        "X":         X,
        "lam":       lam,
        "perc_hex":  perc_hex,
        "abs_hex":   abs_hex,
        "perc_name": perc_name,
        "abs_label": wavelength_to_absorbed_name(lam),
        "verified":  verified,
        "confidence":confidence,
    }

# ── Result rendering ──────────────────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]

    if "Higher confidence" in r["confidence"]:
        st.success(r["confidence"])
    else:
        st.warning(r["confidence"])

    source_note = (
        "Experimentally verified"
        if r["verified"]
        else "Calculated from CIE 1931 + λmax"
    )

    html = f"""
    <div style="border:1px solid #E8E8E8;border-radius:18px;
                padding:24px 28px;background:#FFFFFF;
                box-shadow:0 4px 24px rgba(0,0,0,0.08);
                font-family:sans-serif;">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;
                      text-transform:uppercase;
                      letter-spacing:0.8px;">λMAX PREDICTED</div>
          <div style="font-size:36px;font-weight:900;
                      color:#111;">{r["lam"]:.0f}
            <span style="font-size:16px;color:#999;
                         font-weight:400;">nm</span>
          </div>
          <div style="display:inline-block;padding:3px 10px;
                      border-radius:999px;background:#F0F0F0;
                      font-size:11px;color:#666;margin-top:4px;">
            KFold MAE ≈ 64 nm · Metal-held-out ≈ 101 nm
          </div>
        </div>

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;
                      text-transform:uppercase;
                      letter-spacing:0.8px;">ABSORBED</div>
          <div style="width:56px;height:56px;border-radius:12px;
                      background:{r["abs_hex"]};
                      border:2px solid rgba(0,0,0,0.1);
                      margin:6px 0;"></div>
          <div style="font-size:18px;font-weight:800;
                      color:#111;">{r["abs_label"].upper()}</div>
          <div style="font-size:11px;color:#AAA;
                      font-family:monospace;">{r["abs_hex"]}</div>
        </div>

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;
                      text-transform:uppercase;
                      letter-spacing:0.8px;">PERCEIVED COLOR</div>
          <div style="width:56px;height:56px;border-radius:12px;
                      background:{r["perc_hex"]};
                      border:2px solid rgba(0,0,0,0.1);
                      margin:6px 0;"></div>
          <div style="font-size:18px;font-weight:800;
                      color:#111;">{r["perc_name"].upper()}</div>
          <div style="font-size:11px;color:#AAA;
                      font-family:monospace;">{r["perc_hex"]}</div>
        </div>

      </div>

      <div style="height:1px;background:#EEE;margin:18px 0;">
      </div>

      <div style="color:#AAA;font-size:11px;">
        [{r["metal"]}({r["lig_input"]})]
        <sup>{r["ox_state"]}+</sup> ·
        {r["geometry"]} · {source_note} ·
        91 complexes · ExtraTrees
      </div>
    </div>
    """
    st.components.v1.html(html, height=280, scrolling=False)

    if show_debug:
        with st.expander("Debug: features → model",
                         expanded=False):
            st.dataframe(
                r["X"],
                use_container_width=True,
                hide_index=True
            )