import streamlit as st
from model import load_model, predict_energy

def energy_input_ui():
    st.title("⚡ Energy Consumption Prediction")
    st.markdown("Enter details to predict energy consumption.")

    temperature = st.number_input("Temperature (°C)", -50.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)
    hour_of_day = st.number_input("Hour of Day (0-23)", 0, 23, 12)
    day_of_week = st.number_input("Day of Week (0=Monday, 6=Sunday)", 0, 6, 0)

    input_features = [temperature, humidity, wind_speed, hour_of_day, day_of_week]
    
    model = load_model()
    
    if st.button("Predict Energy Consumption"):
        if model:
            prediction = predict_energy(model, input_features)
            st.success(f"🔋 Predicted Energy Consumption: {prediction:.2f} kWh")
        else:
            st.error("No trained model found. Please train your model first.")
