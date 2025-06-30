# simple_forecast_app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Mock forecasting function (replace with your real model later)
def mock_forecast(last_price, days=30):
    """Generate a realistic mock forecast"""
    base_trend = np.linspace(last_price, last_price * 1.1, days)  # Slight upward trend
    noise = np.random.normal(0, last_price * 0.02, days)  # 2% daily fluctuations
    seasonal = np.sin(np.linspace(0, 3*np.pi, days)) * last_price * 0.05  # Seasonal pattern
    return base_trend + noise + seasonal

# App UI
st.set_page_config(page_title="Tomato Forecast", layout="centered")
st.title("🍅 Simple Tomato Price Forecast")

# User inputs
col1, col2 = st.columns(2)
with col1:
    last_price = st.number_input("Current price (SZL/kg)", min_value=5.0, max_value=100.0, value=25.74)
with col2:
    forecast_days = st.slider("Forecast days", 7, 90, 30)

# Generate and display forecast
if st.button("Generate Forecast"):
    forecast = mock_forecast(last_price, forecast_days)
    dates = [datetime.now() + timedelta(days=i) for i in range(forecast_days)]
    
    # Create dataframe
    forecast_df = pd.DataFrame({
        "Date": dates,
        "Price": forecast.round(2)
    }).set_index("Date")
    
    # Display
    st.subheader(f"{forecast_days}-Day Forecast")
    st.line_chart(forecast_df)
    
    # Show data table
    st.dataframe(forecast_df, height=300)
    
    # Download button
    st.download_button(
        "Download Forecast",
        forecast_df.reset_index().to_csv(index=False),
        f"tomato_forecast_{datetime.now().date()}.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("Note: This demo uses mock data. Replace with your real model for production.")
