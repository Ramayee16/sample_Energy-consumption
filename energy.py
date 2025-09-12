import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("⚡ Energy Consumption Prediction (Linear Regression)")

# Upload training and testing datasets
train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
test_file = st.file_uploader("Upload Testing Dataset (CSV)", type=["csv"])

if train_file is not None and test_file is not None:
    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    st.subheader("📊 Training Data Preview")
    st.write(train_df.head())

    # Assuming last column is target (energy consumption)
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]

    # Split for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_scaled = scaler.transform(test_df)

    # Train model - Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_val_scaled)

    # Metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Plot actual vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y_val, y_pred, alpha=0.6, color="blue")
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    ax.set_xlabel("Actual Energy Consumption")
    ax.set_ylabel("Predicted Energy Consumption")
    ax.set_title("Actual vs Predicted (Linear Regression)")
    st.pyplot(fig)

    # Predict on test dataset
    st.subheader("🔮 Predictions on Test Data")
    test_predictions = model.predict(test_scaled)
    test_df["Predicted_Energy"] = test_predictions
    st.write(test_df.head())

    # Download predictions
    csv_download = test_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")
