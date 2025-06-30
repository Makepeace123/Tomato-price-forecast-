
# tomato_forecast_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load your exact model and scalers
@st.cache_resource
def load_assets():
    model = load_model('tomato_price_model.h5')
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_assets()

# Your exact forecasting function
def forecast_prices(last_features, days=30):
    forecasts = []
    current = last_features.copy()
    for _ in range(days):
        pred = model.predict(current.reshape(1, 1, -1), verbose=0)[0,0]
        forecasts.append(pred)
        current[-1] = pred  # Update target value
    return target_scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

# Streamlit UI
st.title('Tomato Price Forecast')

# Load your latest features (replace with your data loading)
# latest_features = feature_scaler.transform(your_new_data)

if st.button('Generate 30-Day Forecast'):
    forecast = forecast_prices(latest_features)
    forecast_dates = [datetime.now() + timedelta(days=i) for i in range(30)]
    
    st.line_chart(pd.DataFrame({
        'Date': forecast_dates,
        'Price': forecast.flatten()
    }).set_index('Date'))
