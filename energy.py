import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Frontend: Energy Prediction
# ---------------------------

st.set_page_config(page_title="Energy Prediction App", page_icon="⚡", layout="centered")

st.title("⚡ Energy Consumption Prediction")
st.markdown("Enter the details below to predict energy consumption using a trained model.")

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("Input Features")

# Example feature inputs (replace with your actual feature names)
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0, format="%.2f")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, format="%.2f")
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0, format="%.2f")
hour_of_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
day_of_week = st.number_input("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=0)

# ---------------------------
# Load Model
# ---------------------------
try:
    model = joblib.load("energy_model.pkl")  # Make sure the model file exists
except:
    st.warning("No trained model found. Please train a model first!")
    model = None

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Energy Consumption"):
    if model:
        input_features = np.array([[temperature, humidity, wind_speed, hour_of_day, day_of_week]])
        prediction = model.predict(input_features)[0]
        st.success(f"🔋 Estimated Energy Consumption: {prediction:.2f} kWh")
    else:
        st.error("Model not loaded. Cannot make prediction.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using Python & Streamlit")
