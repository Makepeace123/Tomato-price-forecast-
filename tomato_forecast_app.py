# tomato_enhanced_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Cache resources for performance
@st.cache_resource
def load_assets():
    model = load_model('tomato_price_model.h5')
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    last_features = np.load('last_features.npy')  # Saved during training
    return model, feature_scaler, target_scaler, last_features

model, feature_scaler, target_scaler, last_features = load_assets()

# Your forecasting function
def generate_forecast(input_features, days=30):
    forecasts = []
    current = input_features.copy()
    for _ in range(days):
        pred = model.predict(current.reshape(1, 1, -1), verbose=0)[0,0]
        forecasts.append(pred)
        current[-1] = pred
    return target_scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

# Streamlit UI
st.set_page_config(page_title="Tomato AI", layout="wide")
st.title('🌱 Tomato Price Forecaster')

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Latest Market Data")
    
    # Display last known features (edit these as needed)
    st.metric("Last Available Price", 
             f"{target_scaler.inverse_transform(last_features[-1:])[0][0]:.2f} SZL/kg")
    
    days_to_forecast = st.slider("Forecast Days", 7, 90, 30)
    st.caption("Using your trained LSTM model")

with col2:
    if st.button('Generate Forecast', type='primary'):
        with st.spinner('Predicting...'):
            forecast = generate_forecast(last_features, days_to_forecast)
            dates = [datetime.now() + timedelta(days=i) for i in range(1, days_to_forecast+1)]
            
            # Visualization
            chart_data = pd.DataFrame({
                'Date': dates,
                'Forecast': forecast.flatten()
            }).set_index('Date')
            
            st.altair_chart(
                st.altair.Chart(chart_data.reset_index()).mark_line().encode(
                    x='Date:T',
                    y='Forecast:Q'
                ).properties(width=700, title='Price Forecast'),
                use_container_width=True
            )
            
            # Data export
            st.download_button(
                "Export Forecast",
                chart_data.reset_index().to_csv(index=False),
                "tomato_forecast.csv"
            )

st.markdown("---")
st.info("ℹ️ This app uses your trained model with last known market conditions")
