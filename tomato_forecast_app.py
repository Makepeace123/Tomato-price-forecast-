# app.py - Tomato Price Forecasting with Robust CSV Handling
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
DATA_FILE = 'selected_features.csv'

def load_and_prepare_data(file_path):
    """Robust CSV loader that handles formatting issues"""
    try:
        # Try reading with flexible parameters
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            on_bad_lines='warn',  # Skip bad lines instead of failing
            quotechar='"',
            quoting=1,  # QUOTE_MINIMAL
            error_bad_lines=False  # Deprecated in newer pandas, but works in 1.5.3
        )
        
        # Alternative for newer pandas:
        # df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # If empty, try other encodings
        if df.empty:
            for encoding in ['latin1', 'ISO-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='warn')
                    if not df.empty:
                        break
                except:
                    continue
        
        # Verify we got data
        if df.empty:
            st.error("Failed to load CSV - file may be corrupt")
            st.markdown("""
            **Common fixes:**
            1. Open in Excel and save as 'CSV UTF-8 (Comma delimited)'
            2. Ensure all lines have the same number of columns
            3. Check for unquoted commas in text fields
            """)
            st.stop()
            
            # Handle date column
            date_col = None
            if 'Date' in df.columns:
                date_col = pd.to_datetime(df['Date'])
                df = df.drop('Date', axis=1)
            
            # Verify target exists
            if TARGET_COL not in df.columns:
                st.error(f"Column '{TARGET_COL}' not found. Available columns: {list(df.columns)}")
                st.stop()
            
            # Process data
            prices = df[TARGET_COL].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            
            return scaled_prices, scaler, date_col
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error with {encoding}: {str(e)}")
            continue
    
    st.error("Failed to load CSV. Tried encodings: " + ", ".join(encodings))
    st.stop()

# 3. LSTM Model Functions (unchanged)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Forecasting Functions (unchanged)
def generate_forecast(model, last_sequence, forecast_days, scaler):
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
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 Data & Training", "🔮 Forecasting"])
    
    with tab1:
        st.header("Data Preparation")
        
        # Load data
        try:
            scaled_data, scaler, dates = load_and_prepare_data(DATA_FILE)
            st.write(f"Loaded {len(scaled_data)} records")
            
            # Data preview
            if st.checkbox("Show raw data"):
                try:
                    # Display with detected encoding
                    with open(DATA_FILE, 'r', encoding='utf-8') as f:
                        preview = pd.read_csv(f)
                    st.dataframe(preview.head())
                except:
                    # Fallback display
                    st.warning("Couldn't preview with UTF-8. Showing raw content:")
                    with open(DATA_FILE, 'rb') as f:
                        st.text(f.read(1000).decode('ascii', errors='replace'))
            
            # Create sequences
            X, y = create_sequences(scaled_data, SEQ_LENGTH)
            st.write(f"Created {X.shape[0]} training sequences")
            
            # Train model
            if st.button("Train Model"):
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
                    st.success("Model saved!")
                    
                    # Plot training
                    fig, ax = plt.subplots()
                    ax.plot(history.history['loss'])
                    ax.set_title('Training Loss')
                    ax.set_ylabel('Loss')
                    ax.set_xlabel('Epoch')
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
    
    with tab2:
        st.header("Price Forecast")
        
        if st.button("Generate Forecast"):
            try:
                # Load artifacts
                model = load_model('tomato_model.h5')
                scaler = joblib.load('price_scaler.save')
                scaled_data, _, dates = load_and_prepare_data(DATA_FILE)
                
                # Generate forecast
                last_sequence = scaled_data[-SEQ_LENGTH:]
                forecast = generate_forecast(model, last_sequence, FORECAST_DAYS, scaler)
                
                # Create results
                last_date = pd.to_datetime('today') if dates is None else dates.iloc[-1]
                forecast_dates = pd.date_range(start=last_date, periods=FORECAST_DAYS+1)[1:]
                
                results = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price (SZL/kg)': forecast.flatten()
                })
                
                # Display
                st.dataframe(results.style.format({'Forecasted Price (SZL/kg)': '{:.2f}'}))
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results['Date'], results['Forecasted Price (SZL/kg)'], 
                       'r--', marker='o')
                ax.set_title(f'{FORECAST_DAYS}-Day Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Download
                csv = results.to_csv(index=False)
                st.download_button(
                    "Download Forecast",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")

if __name__ == "__main__":
    main()
