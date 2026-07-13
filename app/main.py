from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import streamlit as st
from color_utils import KNOWN_COLORS, spectrum_to_hex, wavelength_to_absorbed_name

ROOT          = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "models" / "best_ml_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_names.pkl"

LIGAND_STRENGTH = {
    'I':0.60,'Br':0.70,'Cl':0.78,'SCN':0.83,'NCS':0.83,
    'F':0.90,'H2O':1.00,'ox':1.10,'acac':1.15,'edta':1.20,
    'NH3':1.25,'dien':1.30,'en':1.40,'bipy':1.50,'phen':1.50,
    'dmg':1.55,'dtc':1.20,'NO2':1.60,'CN':1.70,'CO':1.80,
}
GEOMETRY_CN     = {'octahedral':6,'tetrahedral':4,'square_planar':4}
GEOMETRY_FACTOR = {'octahedral':1.00,'tetrahedral':0.44,'square_planar':1.20}
METAL_Z         = {'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29}
D_ELECTRONS     = {
    ('Ti',3):1,('Ti',2):2,('V',3):2,('V',2):3,
    ('Cr',3):3,('Cr',2):4,('Mn',2):5,('Mn',3):4,
    ('Fe',2):6,('Fe',3):5,('Co',2):7,('Co',3):6,
    ('Ni',2):8,('Ni',3):7,('Cu',2):9,('Cu',1):10,
}

@st.cache_resource
def load_artifacts():
    m  = joblib.load(MODEL_PATH)
    fn = list(joblib.load(FEATURES_PATH))
    return m, fn

model, feature_names = load_artifacts()
@st.cache_resource
def load_meta():
    p = ROOT / "models" / "model_meta.pkl"
    return joblib.load(p) if p.exists() else {}

meta = load_meta()

def build_features(metal, ox, ligand_str, geometry):
    parts = [p.strip() for p in ligand_str.split('+')]
    lig   = sum(LIGAND_STRENGTH.get(p, 1.0) for p in parts) / len(parts)
    gf    = GEOMETRY_FACTOR.get(geometry, 1.0)
    f = {
        'metal_Z':          METAL_Z.get(metal, 26),
        'ox_state':         ox,
        'd_electrons':      D_ELECTRONS.get((metal, ox), 5),
        'ligand_strength':  lig,
        'coord_number':     GEOMETRY_CN.get(geometry, 6),
        'geom_factor':      gf,
        'effective_field':  lig * gf,
        'is_octahedral':    int(geometry == 'octahedral'),
        'is_tetrahedral':   int(geometry == 'tetrahedral'),
        'is_square_planar': int(geometry == 'square_planar'),
    }
    return pd.DataFrame([f]).reindex(columns=feature_names)

# ── страница ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transition Metal Color Predictor",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Transition Metal Color Predictor")
st.caption("ML + Crystal Field Theory → предсказание цвета координационного комплекса")
st.divider()

c1, c2 = st.columns(2)
with c1:
    metal    = st.selectbox("Metal", list(METAL_Z.keys()),
                            index=list(METAL_Z.keys()).index("Co"))
    ox_state = st.number_input("Oxidation state", min_value=1,
                               max_value=9, value=3, step=1)
with c2:
    ligand   = st.selectbox("Ligand", list(LIGAND_STRENGTH.keys()),
                            index=list(LIGAND_STRENGTH.keys()).index("NH3"))
    geometry = st.selectbox("Geometry", list(GEOMETRY_CN.keys()), index=0)

custom = st.text_input("Mixed ligands (e.g. en+NH3)", value="")
show_debug = st.checkbox("Show debug", value=False)
@st.cache_data
def load_raw_df():
    return pd.read_csv(ROOT / "data/raw/complexes_raw.csv")


raw_df = load_raw_df()
if st.button("PREDICT COLOR", type="primary", use_container_width=True):
    lig_input = custom.strip() if custom.strip() else ligand
    X = build_features(metal, int(ox_state), lig_input, geometry)
    pred = float(model.predict(X)[0])
    lam = 1e7 / pred if meta.get("target") == "wavenumber_cm-1" else pred
    seen_metal_ox = ((raw_df["metal"] == metal) & (raw_df["ox_state"] == int(ox_state))).any()
    confidence = "Higher confidence (metal+ox seen in training)" if seen_metal_ox else "Lower confidence (metal+ox not seen in training)"

    key = (metal, int(ox_state), lig_input, geometry)
    if key in KNOWN_COLORS:
        perc_hex, abs_hex, perc_name = KNOWN_COLORS[key]
        verified = True
    else:
        perc_hex, abs_hex, perc_name = spectrum_to_hex(lam, fwhm_nm=120.0)
        verified = False

    st.session_state["result"] = {
        "metal": metal,
        "ox_state": int(ox_state),
        "lig_input": lig_input,
        "geometry": geometry,
        "X": X,
        "lam": lam,
        "perc_hex": perc_hex,
        "abs_hex": abs_hex,
        "perc_name": perc_name,
        "abs_label": wavelength_to_absorbed_name(lam),
        "verified": verified,
        "confidence": confidence,
    }

# ---- Рендер результата (показывается всегда, если уже считали) ----
if "result" in st.session_state:
    
    r = st.session_state["result"]
    if "Higher confidence" in r.get("confidence", ""):
        st.success(r["confidence"])
    else:
        st.warning(r["confidence"])
    source_note = "Experimentally verified color" if r["verified"] else "Calculated from CIE 1931 + λmax"

    html = f"""
    <div style="border:1px solid #E8E8E8;border-radius:18px;padding:24px 28px;
                background:#FFFFFF;box-shadow:0 4px 24px rgba(0,0,0,0.08);
                font-family:sans-serif;">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;">λMAX PREDICTED</div>
          <div style="font-size:36px;font-weight:900;color:#111;">{r["lam"]:.0f}
            <span style="font-size:16px;color:#999;font-weight:400;">nm</span>
          </div>
          <div style="display:inline-block;padding:3px 10px;border-radius:999px;background:#F0F0F0;
                      font-size:11px;color:#666;margin-top:4px;">KFold MAE ≈ 64 nm · Metal-held-out ≈ 101 nm · Metal+ox ≈ 112 nm</div>
        </div>

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;">ABSORBED</div>
          <div style="width:56px;height:56px;border-radius:12px;background:{r["abs_hex"]};
                      border:2px solid rgba(0,0,0,0.1);margin:6px 0;"></div>
          <div style="font-size:18px;font-weight:800;color:#111;">{r["abs_label"].upper()}</div>
          <div style="font-size:11px;color:#AAA;font-family:monospace;">{r["abs_hex"]}</div>
        </div>

        <div style="flex:1;min-width:140px;">
          <div style="color:#999;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;">PERCEIVED (IN SOLUTION)</div>
          <div style="width:56px;height:56px;border-radius:12px;background:{r["perc_hex"]};
                      border:2px solid rgba(0,0,0,0.1);margin:6px 0;"></div>
          <div style="font-size:18px;font-weight:800;color:#111;">{r["perc_name"].upper()}</div>
          <div style="font-size:11px;color:#AAA;font-family:monospace;">{r["perc_hex"]}</div>
        </div>

      </div>

      <div style="height:1px;background:#EEE;margin:18px 0;"></div>

      <div style="color:#AAA;font-size:11px;">
        Complex: [{r["metal"]}({r["lig_input"]})]<sup>{r["ox_state"]}+</sup> · {r["geometry"]} · {source_note} · {r["confidence"]} · 91 complexes · ExtraTrees
      </div>
    </div>
    """
    st.components.v1.html(html, height=280, scrolling=False)

    if show_debug:
        with st.expander("Debug: features → model", expanded=False):
            st.dataframe(r["X"], use_container_width=True, hide_index=True)
            st.code(f'perc={r["perc_hex"]} | abs={r["abs_hex"]} | name={r["perc_name"]}', language="text")
