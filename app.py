import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model
model = load('energy_predictor.joblib')

st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")

st.title("Predict Energy Consumption")
st.markdown("""
Use this tool to estimate **energy consumption** based on:
- Process Temperature
- Environmental Temperature
""")

# Input form
proc_temp = st.number_input("Enter Process Temperature (°C):", min_value=0.0, max_value=200.0, step=0.1)
env_temp = st.number_input("Enter Environmental Temperature (°C):", min_value=-20.0, max_value=60.0, step=0.1)

# Prediction
if st.button("Predict Energy Consumption"):
    input_data = pd.DataFrame([[proc_temp, env_temp]], columns=['ProcTemp', 'EnvTemp'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Energy Consumption: **{prediction:.4f} mKWh**")