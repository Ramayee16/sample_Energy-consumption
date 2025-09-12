import streamlit as st
import numpy as np
import joblib

# Load the trained Energy model
model = joblib.load('linear_regression_model1.joblib')  # Replace with your model path

# Streamlit frontend inputs
st.title("⚡ Energy Consumption Prediction")

temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0)
hour_of_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
day_of_week = st.number_input("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=0)

# Predict button
if st.button("Predict Energy Consumption"):
    # Create input array
    input_data = np.array([[temperature, humidity, wind_speed, hour_of_day, day_of_week]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"🔋 Predicted Energy Consumption: {prediction:.2f} kWh")
