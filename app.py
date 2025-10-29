# app.py — Energy Consumption Predictor (Enhanced)
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime

# -------------------- Page & Theme --------------------
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered",
    menu_items={
        "About": "Open-source IIoT SCADA with AI-enhanced energy forecasting."
    },
)

# Minimal style polish
st.markdown("""
<style>
/* tighten layout a bit */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* nicer success box */
.stAlert {border-radius: 10px;}
/* KPI cards */
.kpi {padding: 1rem; border-radius: 12px; border: 1px solid #eaeaea; background: #fafafa;}
.small {color:#666; font-size:0.9rem;}
.footer {text-align:center; font-size: 0.9rem; color: #666; margin-top: 1.2rem;}
hr {margin: 1.2rem 0;}
</style>
""", unsafe_allow_html=True)

# -------------------- Caching --------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return load(path)

model = load_model("energy_predictor.joblib")  # expects RandomForestRegressor

# -------------------- Sidebar (About / Links) --------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Predict energy consumption from **process** and **environmental** temperatures. "
        "Backed by a Random Forest model trained on real telemetry."
    )
    st.markdown("**Docs & Links**")
    st.markdown("- 📄 Paper: https://bit.ly/Open_Source_SCADA")
    st.markdown("- 💻 Repo: https://github.com/ogatech4real/Energy-Consumption-Predictor")
    st.markdown("- 🌐 Demo (this app): current page")
    st.divider()
    st.caption("Tip: use presets or sliders, then export your prediction history.")

# -------------------- Header --------------------
st.title("Predict Energy Consumption")
st.markdown(
    "This tool estimates **energy consumption (kWh)** using:\n"
    "- Process Temperature (°C)\n"
    "- Environmental Temperature (°C)\n"
)

# -------------------- Presets --------------------
st.subheader("Quick presets")
c1, c2, c3, c4 = st.columns(4)
preset = None
if c1.button("Idle / Cold"):
    preset = (40.0, 5.0)
if c2.button("Nominal"):
    preset = (55.0, 15.0)
if c3.button("Warm Day"):
    preset = (55.0, 30.0)
if c4.button("High Load"):
    preset = (65.0, 15.0)

# -------------------- Inputs --------------------
st.subheader("Inputs")
col_a, col_b = st.columns(2)
if preset:
    pt_default, et_default = preset
else:
    pt_default, et_default = 55.0, 15.0

with col_a:
    proc_temp = st.slider("Process Temperature (°C)", 0.0, 200.0, float(pt_default), 0.1)
    st.number_input("Or type value", min_value=0.0, max_value=200.0, value=float(proc_temp), step=0.1, key="pt_num")
    # keep slider and number input in sync
    proc_temp = st.session_state.pt_num

with col_b:
    env_temp = st.slider("Environmental Temperature (°C)", -20.0, 60.0, float(et_default), 0.1)
    st.number_input("Or type value ", min_value=-20.0, max_value=60.0, value=float(env_temp), step=0.1, key="et_num")
    env_temp = st.session_state.et_num

# -------------------- Predict --------------------
def predict_with_uncertainty(mdl, Xrow: np.ndarray):
    """
    Returns mean prediction and 95% CI using per-tree dispersion.
    Works for RandomForestRegressor (sklearn) if estimators_ is available.
    """
    mean_pred = mdl.predict(Xrow)[0]
    lo = hi = None
    try:
        per_tree = np.array([t.predict(Xrow)[0] for t in mdl.estimators_], dtype=float)
        std = np.std(per_tree)
        lo, hi = mean_pred - 1.96 * std, mean_pred + 1.96 * std
    except Exception:
        pass
    return float(mean_pred), (None if lo is None else float(lo)), (None if hi is None else float(hi))

X = pd.DataFrame([[proc_temp, env_temp]], columns=["ProcTemp", "EnvTemp"])

c_pred = st.button("🚀 Predict Energy Consumption", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

if c_pred:
    yhat, lo, hi = predict_with_uncertainty(model, X.values)
    # Persist in session
    st.session_state.history.append({
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "ProcTemp": proc_temp,
        "EnvTemp": env_temp,
        "Pred_kWh": yhat,
        "CI_low": lo,
        "CI_high": hi
    })

# -------------------- Results UI --------------------
if st.session_state
