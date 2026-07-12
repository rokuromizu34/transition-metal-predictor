"""
Transition Metal Color Predictor - v6
Glowing arc + ONE exact color block
"""

import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Metal Color Predictor", page_icon="🧪", layout="centered")

st.markdown("""
<style>
    .hero-container {
        text-align:center; padding:40px 0 20px 0; position:relative;
    }
    .glow-arc {
        width:400px; height:200px; margin:0 auto;
        border-radius:200px 200px 0 0; background:transparent;
        box-shadow: 0 0 60px rgba(255,107,53,0.3),
                    0 0 120px rgba(255,107,53,0.15),
                    inset 0 0 60px rgba(255,107,53,0.1);
        border-top:2px solid rgba(255,107,53,0.6);
        border-left:1px solid rgba(255,107,53,0.2);
        border-right:1px solid rgba(255,107,53,0.2);
    }
    .hero-title {
        font-size:38px; font-weight:700;
        background:linear-gradient(135deg,#FF6B35,#F7C948);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-top:-80px; position:relative; z-index:2;
    }
    .hero-subtitle { color:#777; font-size:15px; margin-top:10px; }
    .card {
        background:linear-gradient(145deg,#1a1a2e,#16213e);
        border:1px solid #2a2a4a; border-radius:16px; padding:24px; margin:12px 0;
    }
    .result-card {
        background:linear-gradient(145deg,#1a1a1a,#2d1810);
        border:1px solid #FF6B35; border-radius:16px; padding:24px; margin:12px 0;
    }
    .metric-big { font-size:52px; font-weight:700; text-align:center; margin:0; }
    .metric-label {
        font-size:13px; color:#666; text-align:center;
        text-transform:uppercase; letter-spacing:3px;
    }
    .divider {
        height:1px; background:linear-gradient(to right,transparent,#FF6B35,transparent);
        margin:30px 0;
    }
    .info-row {
        display:flex; justify-content:space-between; padding:10px 0;
        border-bottom:1px solid #1a1a2e;
    }
    .info-label { color:#666; font-size:14px; }
    .info-value { color:#eee; font-size:14px; font-weight:500; }
    .footer {
        text-align:center; color:#444; font-size:12px; margin-top:50px;
        padding:20px; border-top:1px solid #1a1a1a;
    }
    .stButton > button {
        background:linear-gradient(135deg,#FF6B35,#e55a2b) !important;
        color:white !important; border:none !important;
        border-radius:12px !important; padding:14px 40px !important;
        font-size:18px !important; font-weight:600 !important;
    }
</style>
""", unsafe_allow_html=True)

model_path = Path(__file__).parent.parent / 'models' / 'best_ml_model.pkl'
model = joblib.load(model_path)

# === EXACT COLORS matching real solutions ===
def exact_observed_color(nm):
    """Returns (name, hex) matching REAL solution colors"""
    if nm < 380: return "colorless", "#E8E8E8"
    elif nm < 410: return "greenish-yellow", "#B8B84B"
    elif nm < 430: return "yellow", "#D4B830"
    elif nm < 450: return "yellow", "#D4B830"
    elif nm < 490: return "yellow-orange", "#D49930"
    elif nm < 510: return "orange", "#CC7722"
    elif nm < 530: return "red", "#A83838"
    elif nm < 545: return "pink", "#C06080"
    elif nm < 560: return "purple", "#6B3FA0"
    elif nm < 580: return "blue-violet", "#4B40A8"
    elif nm < 600: return "blue", "#2860A8"
    elif nm < 630: return "blue-green", "#287878"
    elif nm < 660: return "green", "#388038"
    elif nm < 700: return "green", "#4B8B3B"
    elif nm < 750: return "yellow-green", "#808828"
    elif nm < 800: return "pale green", "#90A860"
    elif nm < 850: return "pale yellow-green", "#A0A850"
    else: return "colorless", "#E8E8E8"

def exact_absorbed_color(nm):
    """Returns (name, hex) of the light being absorbed"""
    if nm < 380: return "UV", "#3a0060"
    elif nm < 430: return "violet", "#6B2D8B"
    elif nm < 470: return "blue", "#2050A0"
    elif nm < 490: return "yellow-orange", "#C8842B"
    elif nm < 500: return "blue-green", "#188070"
    elif nm < 530: return "green", "#288828"
    elif nm < 560: return "yellow-green", "#688820"
    elif nm < 580: return "yellow", "#B8A020"
    elif nm < 610: return "orange", "#C06818"
    elif nm < 660: return "red-orange", "#B03818"
    elif nm < 730: return "red", "#901818"
    elif nm < 780: return "deep red", "#701010"
    else: return "IR", "#400808"

def make_spectrum(predicted_nm):
    def wl_rgb(nm):
        if nm<380: r,g,b=0.6,0.0,0.6
        elif nm<440: r,g,b=-(nm-440)/(440-380),0.0,1.0
        elif nm<490: r,g,b=0.0,(nm-440)/(490-440),1.0
        elif nm<510: r,g,b=0.0,1.0,-(nm-510)/(510-490)
        elif nm<580: r,g,b=(nm-510)/(580-510),1.0,0.0
        elif nm<645: r,g,b=1.0,-(nm-645)/(645-580),0.0
        elif nm<781: r,g,b=1.0,0.0,0.0
        else: r,g,b=0.5,0.0,0.0
        return int(r*255),int(g*255),int(b*255)
    colors=[]
    for wl in range(380,781,2):
        r,g,b=wl_rgb(wl); colors.append(f"rgb({r},{g},{b})")
    gradient=", ".join(colors)
    pos=max(0,min(100,(predicted_nm-380)/400*100))
    return f'''
    <div style="position:relative;height:50px;margin:15px 0;
                border-radius:25px;overflow:hidden;border:1px solid #333;">
        <div style="width:100%;height:100%;
                    background:linear-gradient(to right,{gradient});opacity:0.85;"></div>
        <div style="position:absolute;top:-8px;left:{pos}%;
                    transform:translateX(-50%);font-size:20px;
                    text-shadow:0 0 10px white;">▼</div>
    </div>
    <div style="display:flex;justify-content:space-between;color:#444;
                font-size:11px;margin-top:5px;">
        <span>380 nm</span><span>λmax = {predicted_nm:.0f} nm</span><span>780 nm</span>
    </div>'''

METALS = {
    'Titanium (Ti)':('Ti',22),'Vanadium (V)':('V',23),
    'Chromium (Cr)':('Cr',24),'Manganese (Mn)':('Mn',25),
    'Iron (Fe)':('Fe',26),'Cobalt (Co)':('Co',27),
    'Nickel (Ni)':('Ni',28),'Copper (Cu)':('Cu',29)
}
LIGANDS = {
    'I⁻ iodide':0.60,'Br⁻ bromide':0.70,'Cl⁻ chloride':0.78,
    'SCN⁻ thiocyanate':0.83,'F⁻ fluoride':0.90,'H₂O water':1.00,
    'oxalate':1.10,'acac':1.15,'EDTA':1.20,'NH₃ ammonia':1.25,
    'dien':1.30,'en ethylenediamine':1.40,'bipy bipyridine':1.50,
    'phen phenanthroline':1.50,'NO₂⁻ nitrite':1.60,'CN⁻ cyanide':1.70
}
GEOMETRIES = {
    'Octahedral':('octahedral',6,1.00),
    'Tetrahedral':('tetrahedral',4,0.44),
    'Square Planar':('square_planar',4,1.20)
}
D_ELEC = {
    ('Ti',2):2,('Ti',3):1,('V',2):3,('V',3):2,
    ('Cr',2):4,('Cr',3):3,('Mn',2):5,('Mn',3):4,
    ('Fe',2):6,('Fe',3):5,('Co',2):7,('Co',3):6,
    ('Ni',2):8,('Ni',3):7,('Cu',1):10,('Cu',2):9,
}

# ===== HERO =====
st.markdown("""
<div class="hero-container">
    <div class="glow-arc"></div>
    <p class="hero-title">Transition Metal<br>Color Predictor</p>
    <p class="hero-subtitle">
        Predict absorption wavelength and observed color<br>
        of coordination compounds using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== INPUT =====
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### ⚙️ Parameters")
col1, col2 = st.columns(2)
with col1:
    metal_choice = st.selectbox("Metal", list(METALS.keys()), index=5)
    ox_state = st.selectbox("Oxidation State", [2, 3], index=1)
with col2:
    ligand_choice = st.selectbox("Ligand", list(LIGANDS.keys()), index=9)
    geom_choice = st.selectbox("Geometry", list(GEOMETRIES.keys()))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
predict = st.button("🔮 PREDICT COLOR", type="primary", use_container_width=True)

# ===== RESULTS =====
if predict:
    metal_sym, metal_z = METALS[metal_choice]
    lig_val = LIGANDS[ligand_choice]
    geom_name, cn, gf = GEOMETRIES[geom_choice]
    d_el = D_ELEC.get((metal_sym, ox_state), 5)
    eff = lig_val * gf

    features = pd.DataFrame([{
        'metal_Z':metal_z,'ox_state':ox_state,'d_electrons':d_el,
        'ligand_strength':lig_val,'coord_number':cn,'geom_factor':gf,
        'effective_field':eff,
        'is_octahedral':1 if geom_name=='octahedral' else 0,
        'is_tetrahedral':1 if geom_name=='tetrahedral' else 0,
        'is_square_planar':1 if geom_name=='square_planar' else 0
    }])

    pred = max(300, min(900, model.predict(features)[0]))
    abs_name, abs_hex = exact_absorbed_color(pred)
    obs_name, obs_hex = exact_observed_color(pred)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # WAVELENGTH
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">PREDICTED WAVELENGTH</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-big" style="color:#FF6B35;">{pred:.0f} nm</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#555;font-size:12px;">±40 nm</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # SPECTRUM
    st.markdown("#### Absorption Spectrum")
    st.markdown(make_spectrum(pred), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # === ONE COLOR BLOCK — EXACT COLOR ===
    text_col = "#000" if obs_name in ["yellow","deep yellow","greenish-yellow","orange","colorless","pale green-yellow","yellow-green"] else "#fff"

    st.markdown(f"""
    <div style="text-align:center; margin:20px 0;">
        <p class="metric-label">OBSERVED COLOR OF SOLUTION</p>
        <div style="
            background: {obs_hex};
            width: 100%;
            height: 180px;
            border-radius: 20px;
            margin: 15px 0;
            border: 1px solid #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 40px {obs_hex}44, 0 0 80px {obs_hex}22;
        ">
            <p style="color:{text_col}; font-size:32px; font-weight:700;
                      text-shadow:0 2px 10px rgba(0,0,0,0.4); margin:0;">
                {obs_name}
            </p>
        </div>
        <p style="color:#555; font-size:13px;">
            Absorbs <strong style="color:{abs_hex};">{abs_name}</strong> light at {pred:.0f} nm
            → you see <strong>{obs_name}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # DETAILS
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📋 Complex Details")
    lig_name = ligand_choice.split(' ')[0]

    st.markdown(f"""
    <div class="card">
        <div class="info-row">
            <span class="info-label">Complex</span>
            <span class="info-value">[{metal_sym}({lig_name})]<sup>{ox_state}+</sup></span>
        </div>
        <div class="info-row">
            <span class="info-label">Metal</span>
            <span class="info-value">{metal_choice} (d<sup>{d_el}</sup>)</span>
        </div>
        <div class="info-row">
            <span class="info-label">Ligand field strength</span>
            <span class="info-value">{lig_val:.2f}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Geometry</span>
            <span class="info-value">{geom_choice} (CN = {cn})</span>
        </div>
        <div class="info-row">
            <span class="info-label">Effective field</span>
            <span class="info-value">{eff:.2f}</span>
        </div>
        <div class="info-row" style="border:none;">
            <span class="info-label">Explanation</span>
            <span class="info-value">Absorbs {abs_name} → appears {obs_name}</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <strong>Random Forest Model</strong> · MAE = 40 nm · 91 complexes<br>
    Data: Miessler & Tarr, Inorganic Chemistry (2014)<br><br>
    <a href="https://github.com/rokuromizu34/transition-metal-predictor"
       style="color:#FF6B35;">GitHub</a><br><br>
    Built by Olga · Computational Chemistry × ML
</div>""", unsafe_allow_html=True)