import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from joblib import load

# ---------------- App Config ----------------
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚙️",
    layout="wide"
)

# --------------- Style (subtle, professional) ---------------
st.markdown(
    """
    <style>
        .main .block-container{padding-top:2rem; padding-bottom:2rem; max-width: 1000px;}
        .kpi {background:#F7F9FC; border:1px solid #E6EAF2; border-radius:12px; padding:14px 16px;}
        .caption {color:#6B7280; font-size:0.9rem;}
        .footer {text-align:center; color:#6B7280; font-size:0.9rem; margin-top:1.5rem;}
        .stDownloadButton {margin-top: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return load(path)

model = load_model("energy_predictor.joblib")

# Safety: init session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")
st.sidebar.write(
    "Tweak inputs and explore sensitivity. The model forecasts energy "
    "consumption from **process** and **environmental** temperatures."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Resources")
st.sidebar.markdown(
    "- 📘 Paper summary: https://bit.ly/Open_Source_SCADA\n"
    "- 🧪 Live demo (Streamlit Cloud): your public URL\n"
    "- 💻 Source: your GitHub repo link"
)
st.sidebar.markdown("---")
st.sidebar.subheader("Assumptions")
st.sidebar.caption(
    "• Model trained on 704 records.\n"
    "• Output unit assumed **kWh** (ensure alignment with your training pipeline).\n"
    "• Forecast is point estimate; use intervals as guidance, not guarantees."
)

# ---------------- Header ----------------
st.title("Predict Energy Consumption")
st.markdown(
    "Estimate **energy consumption** from operating conditions.\n"
    "Set temperatures, run a prediction, then explore sensitivity."
)

# ---------------- Input Layout ----------------
col1, col2 = st.columns(2)

with col1:
    proc_temp = st.slider(
        "Process Temperature (°C)",
        min_value=0.0, max_value=200.0, value=55.0, step=0.1,
        help="Typical operating setpoint in your testbed is around 55 °C."
    )
with col2:
    env_temp = st.slider(
        "Environmental Temperature (°C)",
        min_value=-20.0, max_value=60.0, value=15.0, step=0.1,
        help="Ambient temperature near the process equipment."
    )

run = st.button("Predict Energy Consumption", type="primary")

# ---------------- Prediction ----------------
def predict_energy(model, proc_c: float, env_c: float) -> float:
    X = pd.DataFrame([[proc_c, env_c]], columns=["ProcTemp", "EnvTemp"])
    y = model.predict(X)[0]
    return float(y)

if run:
    yhat = predict_energy(model, proc_temp, env_temp)

    # KPI row
    k1, k2, k3 = st.columns([1, 1, 1])
    with k1:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Predicted Energy", f"{yhat:.4f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Process Temp", f"{proc_temp:.1f} °C")
        st.markdown('</div>', unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Ambient Temp", f"{env_temp:.1f} °C")
        st.markdown('</div>', unsafe_allow_html=True)

    # Save to session history
    st.session_state["history"].append(
        {"ProcTemp(°C)": proc_temp, "EnvTemp(°C)": env_temp, "Energy(kWh)": yhat}
    )

# ---------------- Sensitivity (What-if) ----------------
@st.cache_data(show_spinner=False)
def sensitivity_curve(model, proc_c: float, env_min: float, env_max: float, n: int = 41) -> pd.DataFrame:
    env_grid = np.linspace(env_min, env_max, n)
    df = pd.DataFrame({"ProcTemp": np.repeat(proc_c, n), "EnvTemp": env_grid})
    preds = model.predict(df[["ProcTemp", "EnvTemp"]])
    out = pd.DataFrame({"EnvTemp(°C)": env_grid, "Energy(kWh)": preds})
    return out

st.markdown("### Sensitivity: Energy vs. Ambient Temperature")
with st.expander("Show sensitivity chart", expanded=True):
    sens = sensitivity_curve(model, proc_temp, -10, 40, n=41)
    chart = (
        alt.Chart(sens)
        .mark_line(point=True)
        .encode(
            x=alt.X("EnvTemp(°C)", title="Ambient Temperature (°C)"),
            y=alt.Y("Energy(kWh)", title="Predicted Energy (kWh)"),
            tooltip=["EnvTemp(°C)", "Energy(kWh)"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Download button
    csv = sens.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sensitivity data (CSV)",
        data=csv,
        file_name="sensitivity_energy_vs_ambient.csv",
        mime="text/csv"
    )

# ---------------- History ----------------
if st.session_state["history"]:
    st.markdown("### Recent Predictions")
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    <div class='footer'>
        Developed by <strong>Adewale Ogabi</strong> ·
        <a href='https://www.linkedin.com/in/ogabiadewale' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
