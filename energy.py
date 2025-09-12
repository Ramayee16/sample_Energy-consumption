import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Energy Prediction App",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Energy Consumption Prediction")
st.markdown("Enter the details below to predict energy consumption using a trained model.")

# ---------------------------
# Load Model
# ---------------------------
try:
    model = joblib.load("")  # Make sure this file is in the same repo
except FileNotFoundError:
    model = None
    st.warning("No trained model found. Please upload or train your model first!")

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("Input Features")
temperature = st.number_input("Temperature (°C)", -50.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)
hour_of_day = st.number_input("Hour of Day (0-23)", 0, 23, 12)
day_of_week = st.number_input("Day of Week (0=Monday, 6=Sunday)", 0, 6, 0)

input_features = [temperature, humidity, wind_speed, hour_of_day, day_of_week]

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Energy Consumption"):
    if model:
        input_array = np.array([input_features])
        prediction = model.predict(input_array)[0]
        st.success(f"🔋 Predicted Energy Consumption: {prediction:.2f} kWh")
    else:
        st.error("Model not loaded. Cannot make prediction.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using Python & Streamlit")

