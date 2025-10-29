import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from joblib import load

# ---------------- App Config ----------------
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚙️",
    layout="centered"   # single-column feel
)

# --------------- Style (subtle, professional) ---------------
st.markdown(
    """
    <style>
        .main .block-container {padding-top:2rem; padding-bottom:2rem; max-width: 900px;}
        .kpi {background:#F7F9FC; border:1px solid #E6EAF2; border-radius:12px; padding:14px 16px; margin: 0.35rem 0;}
        .caption {color:#6B7280; font-size:0.9rem;}
        .footer {text-align:center; color:#6B7280; font-size:0.9rem; margin-top:1.5rem;}
        .stDownloadButton {margin-top: 0.5rem;}
        .link-list a {text-decoration:none;}
        .pill {display:inline-block; background:#EEF2F7; color:#1F2937; border-radius:999px; padding:4px 10px; font-size:0.85rem; margin-right:6px;}
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

# ---------------- Header + About (previous sidebar content) ----------------
st.title("Predict Energy Consumption")

st.markdown(
    "Estimate **energy consumption** from operating conditions. "
    "Set temperatures, run a prediction, then explore sensitivity."
)

with st.expander("About this tool", expanded=True):
    st.markdown(
        "- **This tool is a predictive web interface developed as part of an AI-enabled, low-cost IIoT SCADA system design**.\n"
        "- It Forecasts energy consumption from **process temperature** and **environmental temperature**.\n"
        "- Built from a Random Forest regression model trained on **704 records**.\n"
        "- It is intended for demo and decision support; treat forecasts as guidance, not guarantees."
    )
    st.markdown(
        '<div class="link-list">'
        'Read more here: <a href="https://bit.ly/Open_Source_SCADA" target="_blank">https://bit.ly/Open_Source_SCADA</a> &nbsp;&nbsp; '
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="pill">Model: Random Forest</span>'
        '<span class="pill">Framework: Scikit-learn</span>'
        '<span class="pill">Unit: kWh</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------- Inputs (single column) ----------------
proc_temp = st.slider(
    "Process Temperature (°C)",
    min_value=0.0, max_value=200.0, value=55.0, step=0.1,
    help="Typical operating setpoint in your testbed is around 55 °C."
)

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

    # KPI cards stacked (single column)
    st.markdown('<div class="kpi"><b>Predicted Energy</b><br><span style="font-size:1.4rem;">'
                f'{yhat:.4f} kWh</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><b>Process Temperature</b><br>'
                f'{proc_temp:.1f} °C</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><b>Ambient Temperature</b><br>'
                f'{env_temp:.1f} °C</div>', unsafe_allow_html=True)

    # Save to session history
    st.session_state["history"].append(
        {"ProcTemp(°C)": proc_temp, "EnvTemp(°C)": env_temp, "Energy(kWh)": yhat}
    )

# ---------------- Sensitivity (What-if) ----------------
# NOTE: Do not pass 'model' to cached function (sklearn estimators are unhashable)
@st.cache_data(show_spinner=False)
def sensitivity_curve(proc_c: float, env_min: float, env_max: float, n: int = 41) -> pd.DataFrame:
    env_grid = np.linspace(env_min, env_max, n)
    df = pd.DataFrame({"ProcTemp": np.repeat(proc_c, n), "EnvTemp": env_grid})
    preds = model.predict(df[["ProcTemp", "EnvTemp"]])  # uses global cached model
    out = pd.DataFrame({"EnvTemp(°C)": env_grid, "Energy(kWh)": preds})
    return out

st.markdown("### Sensitivity: Energy vs. Ambient Temperature")
st.caption("What-if analysis: hold process temperature constant and vary ambient temperature.")
sens = sensitivity_curve(proc_temp, -10.0, 40.0, n=41)
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
    st.caption("Latest results during this session.")
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


