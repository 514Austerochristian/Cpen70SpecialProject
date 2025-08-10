# Start dashboard development
# ===imports===

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import joblib
import json

# Dashboard config
st.set_page_config(
    page_title="Comprehensive Model Water Quality Prediction Comparison Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 150px;
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 0.5rem;
    }
    .metric-trend {
        font-size: 0.8rem;
        color: #777;
    }
    .parameter-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Add src to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global variables and data loading
PROCESSED_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.csv'))

# Define parameters
WATER_PARAMETERS = [
       'Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)',
    'pH Level', 'Dissolved Oxygen (mg/L)'
]

CLIMATE_PARAMETERS = [
    'RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION'
]

POLLUTANT_PARAMETERS = [
    'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 'Phosphate (mg/L)'
]

# Parameter combinations

Waterparams_only = WATER_PARAMETERS
Water_and_Climateparams = WATER_PARAMETERS + CLIMATE_PARAMETERS

# Add the full feature columns for model input (must match training order)
full_feature_columns = [
    'Surface Water Temp (°C)',
    'Middle Water Temp (°C)',
    'Bottom Water Temp (°C)',
    'pH Level',
    'Dissolved Oxygen (mg/L)',
    'RAINFALL',
    'TMAX',
    'TMIN',
    'RH',
    'WIND_SPEED',
    'WIND_DIRECTION',
    'Ammonia (mg/L)',
    'Nitrate-N/Nitrite-N  (mg/L)',
    'Phosphate (mg/L)'
    ]

# Load models for multi-output for enhanced predictions
CNNModel_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'CNNModel.h5')
LSTMModel_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'LSTMModel.h5')
HYBRIDModel_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hybrid_model.h5')

#===Load and preprocess data===
@st.cache_data
def load_and_preprocess_data():
    # Load and preprocess data for different parameter combinations
    try:
        data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.csv'))
        # Perform any necessary preprocessing steps here
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "CNN":
            model = tf.keras.models.load_model(CNNModel_path)
        elif model_name == "LSTM":
            model = tf.keras.models.load_model(LSTMModel_path)
        elif model_name == "HYBRID":
            model = tf.keras.models.load_model(HYBRIDModel_path)
        else:
            raise ValueError("Unknown model name")
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None
    
# Load data and models
data = load_and_preprocess_data()
CNNModel = load_model("CNN")
LSTMModel = load_model("LSTM")
HYBRIDModel = load_model("HYBRID")

"""Enhanced forecasting function for multi-output model (WQI + Pollutant Level)"""
def forecast_multi_output(model, input_data):
    try:
        # Ensure input data is in the correct format
        input_data = np.array(input_data).reshape(1, -1, len(full_feature_columns))
        predictions = model.predict(input_data)
        return predictions[0]  # Return the first prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None 


    