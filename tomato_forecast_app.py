# app.py - Complete Tomato Price Forecasting App
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# 1. Configuration
TARGET_COL = 'Tomato (Round) SZL/1kg'
SEQ_LENGTH = 60  # Number of days to look back
FORECAST_DAYS = 30

# 2. Data Preparation
def load_and_prepare_data(file_path):
    """Load and preprocess the data"""
    df = pd.read_excel(file_path)
    
    # Automatic date handling
    date_col = None
    if 'Date' in df.columns:
        date_col = pd.to_datetime(df['Date'])
        df = df.drop('Date', axis=1)
    
    # Extract target variable
    prices = df[TARGET_COL].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    return scaled_prices, scaler, date_col

# 3. LSTM Model Functions
def create_sequences(data, seq_length):
    """Create input sequences and targets"""
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build the LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Forecasting Functions
def generate_forecast(model, last_sequence, forecast_days, scaler):
    """Generate future predictions"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(forecast_days):
        next_pred = model.predict(current_seq.reshape(1, -1, 1))[0, 0]
        predictions.append(next_pred)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = next_pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 5. Streamlit App
def main():
    st.set_page_config(page_title="Tomato Price Forecast", layout="wide")
    st.title('🍅 Tomato Price Forecasting System')
    
    # Sidebar controls
    st.sidebar.header("Settings")
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 100)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    
    # Main app sections
    tab1, tab2 = st.tabs(["📊 Data & Training", "🔮 Forecasting"])
    
    with tab1:
        st.header("Data Preparation & Model Training")
        
        # Load data
        try:
            scaled_data, scaler, dates = load_and_prepare_data('selected_features.xlsx')
            st.success(f"Data loaded successfully! ({len(scaled_data)} records)")
            
            # Show raw data preview
            if st.checkbox("Show raw data"):
                st.dataframe(pd.read_excel('selected_features.xlsx').head())
            
            # Create sequences
            X, y = create_sequences(scaled_data, SEQ_LENGTH)
            st.write(f"Training sequences created: {X.shape[0]} samples")
            
            # Train model
            if st.button("Train LSTM Model"):
                with st.spinner(f"Training for {epochs} epochs..."):
                    model = build_lstm_model((SEQ_LENGTH, 1))
                    history = model.fit(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[EarlyStopping(monitor='loss', patience=10)],
                        verbose=0
                    )
                    
                    # Save artifacts
                    model.save('tomato_model.h5')
                    joblib.dump(scaler, 'price_scaler.save')
                    st.success("Model trained and saved!")
                    
                    # Plot training
                    fig, ax = plt.subplots()
                    ax.plot(history.history['loss'])
                    ax.set_title('Training Loss Progress')
                    ax.set_ylabel('Loss')
                    ax.set_xlabel('Epoch')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
    
    with tab2:
        st.header("Price Forecasting")
        
        if st.button("Generate 30-Day Forecast"):
            try:
                # Load artifacts
                model = load_model('tomato_model.h5')
                scaler = joblib.load('price_scaler.save')
                scaled_data, _, dates = load_and_prepare_data('selected_features.xlsx')
                
                # Get last sequence
                last_sequence = scaled_data[-SEQ_LENGTH:]
                
                # Generate forecast
                forecast = generate_forecast(model, last_sequence, FORECAST_DAYS, scaler)
                
                # Create results dataframe
                last_date = pd.to_datetime('today') if dates is None else dates.iloc[-1]
                forecast_dates = pd.date_range(start=last_date, periods=FORECAST_DAYS+1)[1:]
                
                results = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price': forecast.flatten()
                })
                
                # Display results
                st.dataframe(results.style.format({'Forecasted Price': '{:.2f}'}))
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results['Date'], results['Forecasted Price'], 
                        marker='o', linestyle='--', color='red')
                ax.set_title(f'{FORECAST_DAYS}-Day Tomato Price Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (SZL/1kg)')
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Download option
                st.download_button(
                    "Download Forecast",
                    data=results.to_csv(index=False),
                    file_name="tomato_forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")

if __name__ == "__main__":
    main()
