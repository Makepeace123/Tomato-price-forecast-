# app.py - Tomato Price Forecasting with Python 3.9 Compatibility
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =============================================
# 1. Configuration
# =============================================
TARGET_COL = 'Tomato (Round) SZL/1kg'
SEQ_LENGTH = 60  # Number of days to look back
FORECAST_DAYS = 30
MODEL_FILE = 'tomato_model.h5'
SCALER_FILE = 'price_scaler.save'
DATA_FILE = 'selected_features.xlsx'

# =============================================
# 2. Dependency Check
# =============================================
def check_dependencies():
    """Verify all required packages are installed"""
    required = {
        'streamlit': '1.29.0',
        'tensorflow': '2.10.1',
        'pandas': '1.5.3',
        'openpyxl': '3.1.2',
        'scikit-learn': '1.0.2'
    }
    
    missing = []
    for pkg, ver in required.items():
        try:
            m = __import__(pkg)
            if m.__version__ != ver:
                missing.append(f"{pkg}=={ver}")
        except ImportError:
            missing.append(f"{pkg}=={ver}")
    
    if missing:
        st.error(f"Missing dependencies. Run:\n\n`pip install {' '.join(missing)}`")
        st.stop()

# =============================================
# 3. Data Loading and Preparation
# =============================================
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_excel(DATA_FILE, engine='openpyxl')
        
        # Handle date column
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            df = df.drop('Date', axis=1)
        else:
            dates = None
            
        # Verify target column exists
        if TARGET_COL not in df.columns:
            st.error(f"Target column '{TARGET_COL}' not found in dataset")
            st.stop()
            
        prices = df[TARGET_COL].values.reshape(-1, 1)
        return prices, dates
    
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()

# =============================================
# 4. Model Functions
# =============================================
def create_sequences(data, seq_length):
    """Create input sequences for LSTM"""
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def build_model():
    """Construct LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train):
    """Train the LSTM model"""
    model = build_model()
    history = model.fit(
        X_train, 
        y_train,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='loss', patience=10)],
        verbose=0
    )
    return model, history

# =============================================
# 5. Forecasting Functions
# =============================================
def generate_forecast(model, last_sequence, scaler):
    """Generate future predictions"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(FORECAST_DAYS):
        next_pred = model.predict(current_seq.reshape(1, -1, 1))[0, 0]
        predictions.append(next_pred)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = next_pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# =============================================
# 6. Streamlit App Interface
# =============================================
def main():
    # App configuration
    st.set_page_config(page_title="Tomato Price Forecast", layout="wide")
    st.title("🍅 Tomato Price Forecasting System")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # ======================
    # Sidebar Controls
    # ======================
    with st.sidebar:
        st.header("Settings")
        epochs = st.slider("Training Epochs", 10, 200, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        
        st.markdown("---")
        st.markdown("**Requirements:**")
        st.code("Python 3.9\nTensorFlow 2.10.1")
    
    # ======================
    # Main App Sections
    # ======================
    tab1, tab2 = st.tabs(["📊 Data & Training", "🔮 Forecasting"])
    
    with tab1:
        st.header("Data Preparation & Model Training")
        
        # Load and prepare data
        prices, dates = load_data()
        st.success(f"Data loaded successfully! ({len(prices)} records)")
        
        # Data preview
        if st.checkbox("Show raw data preview"):
            df = pd.read_excel(DATA_FILE, engine='openpyxl')
            st.dataframe(df.head())
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = create_sequences(scaled_prices, SEQ_LENGTH)
        st.write(f"Training sequences created: {X.shape[0]} samples")
        
        # Train model button
        if st.button("Train LSTM Model"):
            with st.spinner(f"Training model for {epochs} epochs..."):
                model, history = train_model(X, y)
                
                # Save artifacts
                model.save(MODEL_FILE)
                joblib.dump(scaler, SCALER_FILE)
                st.session_state.model_trained = True
                st.success("Model trained and saved!")
                
                # Plot training history
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'])
                ax.set_title('Model Training Progress')
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epoch')
                st.pyplot(fig)
    
    with tab2:
        st.header("Price Forecasting")
        
        if st.button("Generate Forecast"):
            if not os.path.exists(MODEL_FILE):
                st.error("No trained model found. Please train the model first.")
                st.stop()
                
            try:
                # Load artifacts
                model = load_model(MODEL_FILE)
                scaler = joblib.load(SCALER_FILE)
                
                # Get last sequence
                prices, _ = load_data()
                scaled_prices = scaler.transform(prices)
                last_sequence = scaled_prices[-SEQ_LENGTH:]
                
                # Generate forecast
                forecast = generate_forecast(model, last_sequence, scaler)
                
                # Create results dataframe
                last_date = pd.to_datetime('today') if dates is None else dates.iloc[-1]
                forecast_dates = pd.date_range(start=last_date, periods=FORECAST_DAYS+1)[1:]
                
                results = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price (SZL/kg)': forecast.flatten()
                })
                
                # Display results
                st.subheader(f"{FORECAST_DAYS}-Day Price Forecast")
                st.dataframe(results.style.format({'Forecasted Price (SZL/kg)': '{:.2f}'}))
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results['Date'], results['Forecasted Price (SZL/kg)'], 
                       marker='o', linestyle='--', color='red')
                ax.set_title('Tomato Price Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (SZL/kg)')
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Download option
                csv = results.to_csv(index=False)
                st.download_button(
                    "Download Forecast",
                    data=csv,
                    file_name="tomato_price_forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")

# =============================================
# 7. Run the App
# =============================================
if __name__ == "__main__":
    # Set environment variables
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check dependencies first
    check_dependencies()
    
    # Run main app
    main()
