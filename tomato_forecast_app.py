import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pickle  # Alternative to joblib
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Add this line first

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
# Cache resources
@st.cache_resource
def load_assets():
    try:
        model = load_model('tomato_price_model.h5')
        with open('feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        last_features = np.load('last_features.npy', allow_pickle=True)
        return model, feature_scaler, target_scaler, last_features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

model, feature_scaler, target_scaler, last_features = load_assets()

if model is None:
    st.stop()

# Forecast function
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
st.title('🍅 Tomato Price Forecaster')

if st.button('Generate 30-Day Forecast', type='primary'):
    with st.spinner('Generating forecast...'):
        forecast = generate_forecast(last_features)
        dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
        
        # Show results
        st.subheader('Price Forecast')
        chart_data = pd.DataFrame({
            'Date': dates,
            'Price (SZL/kg)': forecast.flatten()
        }).set_index('Date')
        st.line_chart(chart_data)
        
        # Download button
        st.download_button(
            "Download Forecast",
            chart_data.reset_index().to_csv(index=False),
            "tomato_forecast.csv"
        )

st.markdown("---")
st.info("ℹ️ Using your trained LSTM model with last known market conditions")
