import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib

# Set the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Page config
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
        <style>
            .main {
                background-color: #f0f2f5;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stTextInput>div>input {
                border-radius: 5px;
                border: 1px solid #ccc;
                padding: 10px;
            }
            .stTextInput>div>input:focus {
                border-color: #4CAF50;
            }
        </style>
    """,
    unsafe_allow_html=True
)

# Add the title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("""
    This dashboard allows you to predict stock prices using a pre-trained LSTM model.
    You can input the stock symbol and select the prediction date range.
""")    

# Add the src directory to the system path
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Global variables and Data Loading
MODEL_PATH = os.path.join(parent_dir, 'models', 'lstm_model.h5')
SCALER_PATH = os.path.join(parent_dir, 'models', 'scaler.pkl')
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure the model is trained and saved.")
    st.stop()
if not os.path.exists(SCALER_PATH):
    st.error("Scaler file not found. Please ensure the scaler is trained and saved.")
    st.stop()
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Function to load stock data
def load_stock_data(symbol, start_date, end_date):
    try:
        df = pd.read_csv(f'data/{symbol}.csv', parse_dates=['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        return df
    except FileNotFoundError:
        st.error(f"Data for {symbol} not found. Please check the symbol and try again.")
        return None

# Define parameters for user input
Water_parameters = {
    'symbol': st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL"),
    'start_date': st.date_input("Start Date:", datetime.now() - timedelta(days=365)),
    'end_date': st.date_input("End Date:", datetime.now()),
    'prediction_days': st.slider("Prediction Days:", 1, 30, 7)
}

Water_and_Climatic_parameters = {
    'symbol': Water_parameters['symbol'],
    'start_date': Water_parameters['start_date'],
    'end_date': Water_parameters['end_date'],
    'prediction_days': Water_parameters['prediction_days']
}   

# Parameters for user input
st.sidebar.header("User Input Parameters")
symbol = st.sidebar.text_input("Stock Symbol", Water_parameters['symbol'])
start_date = st.sidebar.date_input("Start Date", Water_parameters['start_date'])
end_date = st.sidebar.date_input("End Date", Water_parameters['end_date'])
prediction_days = st.sidebar.slider("Prediction Days", 1, 30, Water_parameters['prediction_days']) 
symbol = st.sidebar.text_input("Stock Symbol", Water_and_Climatic_parameters['symbol'])
start_date = st.sidebar.date_input("Start Date", Water_and_Climatic_parameters['start_date'])
end_date = st.sidebar.date_input("End Date", Water_and_Climatic_parameters['end_date'])
prediction_days = st.sidebar.slider("Prediction Days", 1, 30, Water_and_Climatic_parameters['prediction_days'])

# Add the full feature columns for model input(must match the model's training data)
feature_columns = ['Rainfall', 
                   'Temperature', 
                   'Humidity', 
                   'WindSpeed', 
                   'WindDirection', 
                   'Temperature_Max', 
                   'Temperature_Min', 
                   'Surface_Water_Temperature (°C)',
                   'Bottom_Water_Temperature (°C)',
                   'Middle_Water_Temperature (°C)',
                   'Dissolved_Oxygen (mg/L)',
                   'pH',
                   'Turbidity (NTU)',
                   'Ammonia (mg/L)',
                   'Phosphate (mg/L)',
                   'Nitrate-N/Nitrite-N (mg/L)', 
                   'WQI'
                   ]



