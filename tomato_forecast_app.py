import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import requests
import os

# Demo configuration
st.set_page_config(page_title="Tomato Price Forecast Demo", layout="wide")

# =============================================
# MODIFIED KEEP-ALIVE SYSTEM (FULLY WORKING VERSION)
# =============================================
def keep_alive():
    """Background thread to prevent app sleeping"""
    while True:
        try:
            app_url = os.getenv('STREAMLIT_SERVER_BASE_URL', 'http://localhost:8501')
            requests.get(f"{app_url}/?keepalive=1", timeout=5)
            time.sleep(240)  # Ping every 4 minutes
        except:
            time.sleep(60)

if 'keep_alive_started' not in st.session_state:
    st.session_state.keep_alive_started = True
    t = threading.Thread(target=keep_alive, daemon=True)
    t.start()

# Check for keepalive ping AFTER Streamlit is initialized
if hasattr(st, 'query_params') and st.query_params.get('keepalive'):
    st.write("")  # Empty response
    st.stop()

# Client-side auto-refresh
html("""
<script>
setTimeout(function(){ location.reload(); }, 5*60*1000);
</script>
""", height=0, width=0)
# =============================================

st.title("🍅 Tomato Price Forecasting Demo")

# Pre-generated demo data
def generate_demo_data():
    dates = pd.date_range(start="2025-01-01", periods=30)
    forecast = np.linspace(36, 32, 30) + np.random.normal(0, 0.5, 30)
    lower = forecast - 0.7 + np.random.random(30)*0.3
    upper = forecast + 0.7 + np.random.random(30)*0.3
    actual = forecast + np.random.normal(0, 0.3, 30)
    
    return pd.DataFrame({
        "Date": dates,
        "Forecasted_Price": forecast.round(2),
        "Lower_Bound": lower.round(2),
        "Upper_Bound": upper.round(2),
        "Actual_Price": actual.round(2)
    })

# Main demo function
def run_demo():
    st.sidebar.header("Demo Controls")
    st.sidebar.info("This is a simulated forecast showing how the app would work with real data")
    
    # Generate data
    demo_data = generate_demo_data()
    
    # Show raw data
    with st.expander("View Raw Forecast Data"):
        st.dataframe(demo_data.style.format({
            "Forecasted_Price": "{:.2f}",
            "Lower_Bound": "{:.2f}",
            "Upper_Bound": "{:.2f}",
            "Actual_Price": "{:.2f}"
        }))
    
    # Main visualization
    st.subheader("30-Day Price Forecast")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot confidence interval
    ax.fill_between(
        demo_data["Date"],
        demo_data["Lower_Bound"],
        demo_data["Upper_Bound"],
        color="orange",
        alpha=0.2,
        label="Confidence Interval"
    )
    
    # Plot lines
    ax.plot(
        demo_data["Date"],
        demo_data["Forecasted_Price"],
        label="Forecast",
        color="red",
        marker="o"
    )
    
    ax.plot(
        demo_data["Date"],
        demo_data["Actual_Price"],
        label="Actual (Simulated)",
        color="blue",
        linestyle="--"
    )
    
    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (SZL/kg)")
    ax.set_title("Tomato Price Forecast with Confidence Bounds")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price", 
            f"{demo_data['Actual_Price'].iloc[0]:.2f} SZL/kg",
            f"{(demo_data['Actual_Price'].iloc[0] - demo_data['Actual_Price'].iloc[-1]):.2f} vs last month"
        )
    with col2:
        st.metric(
            "30-Day Forecast Avg",
            f"{demo_data['Forecasted_Price'].mean():.2f} SZL/kg"
        )
    with col3:
        st.metric(
            "Price Volatility",
            f"±{(demo_data['Upper_Bound'] - demo_data['Lower_Bound']).mean():.2f} SZL/kg"
        )
    
    # Download button
    csv = demo_data.to_csv(index=False)
    st.download_button(
        "Download Demo Data",
        data=csv,
        file_name="tomato_price_forecast_demo.csv",
        mime="text/csv"
    )

# Run the demo
if __name__ == "__main__":
    run_demo()
