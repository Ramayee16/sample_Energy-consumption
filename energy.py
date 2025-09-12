import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

st.title("Energy Consumption Prediction")

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(df.head())

    # 2️⃣ Select features and target
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column (Energy Consumption)", all_columns)
    feature_columns = st.multiselect("Select Feature Columns", [col for col in all_columns if col != target_column])

    if st.button("Train Model"):
        # Prepare data
        X = df[feature_columns]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.success(f"Model trained! Mean Squared Error: {mse:.2f}")
        
        # Save the trained model
        joblib.dump(model, "energy_model.pkl")
        st.info("Trained model saved as 'energy_model.pkl'")

# 3️⃣ Predict energy consumption for new data
st.subheader("Predict Energy Consumption")
if st.checkbox("Use the trained model to predict"):
    # Load the model
    try:
        model = joblib.load("energy_model.pkl")
    except:
        st.error("No trained model found. Please train a model first!")
    
    if model:
        # Input values for prediction
        input_data = []
        for feature in model.feature_names_in_:
            val = st.number_input(f"Enter value for {feature}", value=0.0)
            input_data.append(val)
        
        if st.button("Predict Energy Consumption"):
            input_array = np.array([input_data])
            prediction = model.predict(input_array)[0]
            st.success(f"🔋 Predicted Energy Consumption: {prediction:.2f}")
