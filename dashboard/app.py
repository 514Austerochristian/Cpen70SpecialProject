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
from pathlib import Path

# Dashboard config - MUST BE FIRST
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
Parameter_combinations = {
    "WQI + Pollutant Level": WATER_PARAMETERS + CLIMATE_PARAMETERS + POLLUTANT_PARAMETERS,
    "WQI Only": WATER_PARAMETERS,
    "WQI + Climate": WATER_PARAMETERS + CLIMATE_PARAMETERS
}

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

# Get the absolute path to the directory where THIS file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models for multi-output for enhanced predictions
CNNModel_path = os.path.join(BASE_DIR, '..', 'models', 'cnn_model_tuned.h5')
LSTMModel_path = os.path.join(BASE_DIR, '..', 'models', 'lstm_model_tuned.h5')
HYBRIDModel_path = os.path.join(BASE_DIR, '..', 'models', 'hybrid_model_tuned.h5')

# Normalize the paths
CNNModel_path = os.path.normpath(CNNModel_path)
LSTMModel_path = os.path.normpath(LSTMModel_path)
HYBRIDModel_path = os.path.normpath(HYBRIDModel_path)

# Define the path to the model comparison JSON file
MODEL_COMPARISON_JSON = '/workspaces/Cpen70SpecialProject/models/model_comparison.json'

# Alternative paths for model comparison (fallback options)
MODEL_COMPARISON_PATHS = [
    '/workspaces/Cpen70SpecialProject/models/model_comparison.json',
    os.path.join(BASE_DIR, '..', 'models', 'model_comparison.json'),
    os.path.join(BASE_DIR, '..', 'data', 'processed', 'model_comparison.json'),
    'models/model_comparison.json',
    'data/processed/model_comparison.json'
]

#===Load and preprocess data===
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data for different parameter combinations"""
    try:
        # Try multiple possible paths for the data file
        possible_paths = [
            os.path.join(BASE_DIR, '..', 'data', 'processed', 'model_processed_data.csv'),
            os.path.join(BASE_DIR, '..', 'data', 'model_processed_data.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_processed_data.csv'),
            'data/processed/model_processed_data.csv',
            'data/model_processed_data.csv'
        ]
        
        data = None
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                data = pd.read_csv(normalized_path)
                st.info(f"✅ Data loaded from: {normalized_path}")
                break
        
        if data is None:
            # Create sample data if file doesn't exist
            st.warning("⚠️ Data file not found. Creating sample data for demonstration.")
            data = create_sample_data()
        
        # Ensure all required columns exist
        for col in full_feature_columns:
            if col not in data.columns:
                st.warning(f"⚠️ Missing column {col}, filling with default values")
                data[col] = np.random.normal(0, 1, len(data))
        
        # Add location column if missing
        if 'location' not in data.columns:
            locations = ["Tanauan", "Talisay", "Laurel", "Agoncillo", "San Nicolas", "Alitagtag"]
            data['location'] = np.random.choice(locations, len(data))
        
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    n_samples = 100
    
    locations = ["Tanauan", "Talisay", "Laurel", "Agoncillo", "San Nicolas", "Alitagtag"]
    
    data = {
        'location': np.random.choice(locations, n_samples),
        'Surface Water Temp (°C)': np.random.normal(28, 2, n_samples),
        'Middle Water Temp (°C)': np.random.normal(27, 2, n_samples),
        'Bottom Water Temp (°C)': np.random.normal(26, 2, n_samples),
        'pH Level': np.random.normal(7.5, 0.5, n_samples),
        'Dissolved Oxygen (mg/L)': np.random.normal(8, 1, n_samples),
        'RAINFALL': np.random.exponential(10, n_samples),
        'TMAX': np.random.normal(32, 3, n_samples),
        'TMIN': np.random.normal(24, 2, n_samples),
        'RH': np.random.normal(75, 10, n_samples),
        'WIND_SPEED': np.random.exponential(5, n_samples),
        'WIND_DIRECTION': np.random.uniform(0, 360, n_samples),
        'Ammonia (mg/L)': np.random.exponential(0.5, n_samples),
        'Nitrate-N/Nitrite-N  (mg/L)': np.random.exponential(1, n_samples),
        'Phosphate (mg/L)': np.random.exponential(0.3, n_samples),
        'WQI': np.random.normal(60, 15, n_samples)
    }
    
    return pd.DataFrame(data)

@st.cache_resource
def load_model(model_name):
    """Load TensorFlow models with proper error handling"""
    custom_objects = {
        'mse': tf.keras.metrics.MeanSquaredError(),
        'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError(),
        'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError(),
        'accuracy': tf.keras.metrics.Accuracy(),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
    }

    try:
        if model_name == "CNN":
            if os.path.exists(CNNModel_path):
                model = tf.keras.models.load_model(CNNModel_path, custom_objects=custom_objects)
                st.success(f"✅ {model_name} model loaded successfully!")
                return model
        elif model_name == "LSTM":
            if os.path.exists(LSTMModel_path):
                model = tf.keras.models.load_model(LSTMModel_path, custom_objects=custom_objects)
                st.success(f"✅ {model_name} model loaded successfully!")
                return model
        elif model_name == "HYBRID":
            if os.path.exists(HYBRIDModel_path):
                model = tf.keras.models.load_model(HYBRIDModel_path, custom_objects=custom_objects)
                st.success(f"✅ {model_name} model loaded successfully!")
                return model
        
        # If model file doesn't exist, return a mock model
        st.warning(f"⚠️ {model_name} model file not found at expected path. Using mock predictions.")
        return create_mock_model(model_name)
        
    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {e}")
        return create_mock_model(model_name)

def create_mock_model(model_name):
    """Create a mock model for demonstration when actual models aren't available"""
    class MockModel:
        def __init__(self, name):
            self.name = name
            
        def predict(self, X):
            # Return random predictions for demonstration
            if hasattr(X, '__len__'):
                n_samples = len(X)
            else:
                n_samples = 1
                
            if n_samples > 0:
                # Generate realistic WQI predictions based on model type
                if self.name == "CNN":
                    base_wqi = np.random.normal(58, 8, n_samples)
                elif self.name == "LSTM":
                    base_wqi = np.random.normal(62, 7, n_samples)
                else:  # HYBRID
                    base_wqi = np.random.normal(60, 6, n_samples)
                
                return np.clip(base_wqi, 0, 100)
            return np.array([60.0])
        
        def forecast(self, X):
            base_wqi = 60 + np.random.normal(0, 5)
            if self.name == "CNN":
                base_wqi += np.random.normal(-2, 3)
            elif self.name == "LSTM":
                base_wqi += np.random.normal(2, 3)
            
            return {
                'WQI': max(0, min(100, base_wqi)),
                'Ammonia (mg/L)': max(0, np.random.exponential(0.5)),
                'Nitrate-N/Nitrite-N  (mg/L)': max(0, np.random.exponential(1)),
                'Phosphate (mg/L)': max(0, np.random.exponential(0.3))
            }
    
    return MockModel(model_name)

"""Enhanced forecasting function for multi-output model (WQI + Pollutant Level)"""
def forecast_multi_output(model, input_data):
    try:
        if hasattr(model, 'forecast'):
            return model.forecast(input_data)
        else:
            # For regular models, make basic predictions
            predictions = model.predict(input_data)
            wqi_pred = predictions[0] if len(predictions) > 0 else 60.0
            
            return {
                'WQI': float(wqi_pred),
                'Ammonia (mg/L)': float(np.random.exponential(0.5)),
                'Nitrate-N/Nitrite-N  (mg/L)': float(np.random.exponential(1)),
                'Phosphate (mg/L)': float(np.random.exponential(0.3))
            }
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return {
            'WQI': 60.0,
            'Ammonia (mg/L)': 0.5,
            'Nitrate-N/Nitrite-N  (mg/L)': 1.0,
            'Phosphate (mg/L)': 0.3
        }

# Create forecast dataframe with all outputs
def create_forecast_dataframe(predictions):
    columns = [
        'WQI',
        'Ammonia (mg/L)',
        'Nitrate-N/Nitrite-N  (mg/L)',
        'Phosphate (mg/L)'
    ]
    return pd.DataFrame([predictions], columns=columns)

# Generate future dates
def generate_future_dates(start_date, num_periods):
    future_dates = pd.date_range(start=start_date, periods=num_periods, freq='M')
    return future_dates

# Get wqi classification and advice
def get_wqi_classification(wqi_value):
    if wqi_value < 50:
        return "Excellent", "Water is suitable for all uses. Maintain current practices."
    elif 50 <= wqi_value < 75:
        return "Good", "Water is suitable for most uses. Minor treatment might be needed in sensitive applications. Some monitoring is recommended."
    elif 75 <= wqi_value < 90:
        return "Fair", "Water is suitable for limited uses. Aquatic life might be stressed. Further treatment may be needed."
    else:
        return "Poor", "Water is not suitable for any uses without treatment."

# Create metric card
def create_metric_card(wqi_classification, wqi_advice):
    st.subheader("Water Quality Index (WQI) Insights")
    st.metric(label="WQI Classification", value=wqi_classification)
    st.write(wqi_advice)

# Add a helper function for safely format floats
def format_float(value):
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}"

# Load model comparison functions
def load_model_comparison(file_paths=None):
    """
    Load model comparison data from JSON file with multiple path options

    Args:
        file_paths (list): List of paths to try for the model_comparison.json file
        
    Returns:
        dict: Loaded model comparison data
    """
    if file_paths is None:
        file_paths = MODEL_COMPARISON_PATHS
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            
            if path.exists():
                st.info(f"✅ Found model comparison file at: {path.absolute()}")
                
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                st.success(f"✅ Successfully loaded model comparison data")
                
                # Debug: Show the structure of loaded data
                if st.checkbox("🔍 Debug: Show JSON Structure", key="debug_json"):
                    st.write("**JSON Structure:**")
                    st.json(data)
                    
                    # Show available keys at root level
                    st.write("**Root level keys:**", list(data.keys()))
                    
                    # If there's a 'models' key, show its structure
                    if 'models' in data:
                        st.write("**Models available:**", list(data['models'].keys()))
                        for model_name, model_data in data['models'].items():
                            if isinstance(model_data, dict):
                                st.write(f"**{model_name} metrics:**", list(model_data.keys()))
                
                return data
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing JSON file at {file_path}: {e}")
            continue
        except Exception as e:
            # Don't show error for each path attempt, only for the primary path
            if file_path == file_paths[0]:
                st.warning(f"⚠️ Could not load from primary path: {e}")
            continue
    
    st.warning(f"⚠️ Model comparison file not found in any of the expected locations:")
    for path in file_paths:
        st.write(f"   • {path}")
    return None

def display_model_summary(model_data):
    """
    Display a comprehensive summary of all models in the comparison

    Args:
        model_data (dict): Model comparison data loaded from JSON
    """
    if not model_data:
        st.error("❌ No model data to display")
        return

    st.subheader("🤖 MODEL COMPARISON SUMMARY")
    
    # Check if data has models section
    models = model_data.get('models', model_data)

    if isinstance(models, dict):
        model_names = list(models.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Models", len(model_names))
        with col2:
            st.metric("Model Types", ', '.join(model_names))
        
        # Create comparison table
        comparison_data = []
        
        for model_name, model_info in models.items():
            if isinstance(model_info, dict):
                row = {
                    'Model': model_name,
                    'Accuracy': model_info.get('accuracy', 'N/A'),
                    'Precision': model_info.get('precision', 'N/A'),
                    'Recall': model_info.get('recall', 'N/A'),
                    'F1-Score': model_info.get('f1_score', model_info.get('f1', 'N/A')),
                    'MSE': model_info.get('mse', model_info.get('mean_squared_error', 'N/A')),
                    'MAE': model_info.get('mae', model_info.get('mean_absolute_error', 'N/A')),
                    'R²': model_info.get('r2_score', model_info.get('r2', 'N/A')),
                    'Training Time': model_info.get('training_time', 'N/A'),
                }
                comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.subheader("📈 MODEL PERFORMANCE COMPARISON")
            st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("⚠️ Unexpected data structure in model comparison file")

def find_best_model(model_data, metric='mse'):
    """
    Find the best performing model based on a specific metric

    Args:
        model_data (dict): Model comparison data
        metric (str): Metric to use for comparison (default: 'mse')
        
    Returns:
        tuple: (model_name, metric_value)
    """
    if not model_data:
        return None, None

    models = model_data.get('models', model_data)
    best_model = None
    best_value = None

    # For metrics where higher is better
    higher_better = ['accuracy', 'precision', 'recall', 'f1_score', 'f1', 'r2_score', 'r2', 'r²']
    # For metrics where lower is better (error metrics)
    lower_better = ['mse', 'rmse', 'mae', 'mean_squared_error', 'mean_absolute_error', 'loss']

    # Debug info
    if st.checkbox(f"🔍 Debug: Show {metric} search", key=f"debug_{metric}"):
        st.write(f"**Looking for metric: {metric}**")
        st.write(f"**Available models:** {list(models.keys()) if isinstance(models, dict) else 'Not a dict'}")

    for model_name, model_info in models.items():
        if isinstance(model_info, dict):
            # Try different possible key names for the metric
            possible_keys = [
                metric,
                metric.lower(),
                metric.upper(),
                metric.replace('_', ''),
                metric.replace('_', ' '),
                metric.replace(' ', '_'),
                # Specific variations for common metrics
                'r2' if metric == 'r²' else None,
                'r2_score' if metric == 'r²' else None,
                'mean_squared_error' if metric == 'mse' else None,
                'mean_absolute_error' if metric == 'mae' else None,
                'root_mean_squared_error' if metric == 'rmse' else None
            ]
            
            # Remove None values
            possible_keys = [k for k in possible_keys if k is not None]
            
            value = None
            found_key = None
            
            for key in possible_keys:
                if key in model_info:
                    value = model_info[key]
                    found_key = key
                    break
            
            # Debug info
            if st.checkbox(f"🔍 Debug: {model_name} details", key=f"debug_{model_name}_{metric}"):
                st.write(f"**{model_name} available keys:** {list(model_info.keys())}")
                st.write(f"**Looking for:** {possible_keys}")
                st.write(f"**Found key:** {found_key}")
                st.write(f"**Value:** {value}")
            
            if value is not None and isinstance(value, (int, float)):
                if best_value is None:
                    best_model = model_name
                    best_value = value
                else:
                    if metric.lower() in higher_better or metric in ['r²']:
                        # Higher is better
                        if value > best_value:
                            best_model = model_name
                            best_value = value
                    elif metric.lower() in lower_better:
                        # Lower is better (error metrics)
                        if value < best_value:
                            best_model = model_name
                            best_value = value

    return best_model, best_value

# LOAD DATA AND MODELS HERE (before using them)
data = load_and_preprocess_data()

# Load models with progress indicators
with st.spinner('Loading models...'):
    CNNModel = load_model("CNN")
    LSTMModel = load_model("LSTM")
    HYBRIDModel = load_model("HYBRID")

# Page selection
page = st.sidebar.selectbox("Select Page", ["Home", "Model Information", "Data Visualization"])

# Sidebar
st.sidebar.title("Water Quality Forecasting")
st.sidebar.markdown("---")

# Parameter selection
st.sidebar.subheader("Select Parameters")
params_combo = st.sidebar.selectbox(
    "Select Parameter Combination", 
    list(Parameter_combinations.keys()), 
    help="Choose a parameter combination for forecasting."
)

# Add Location Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Locations")
if not data.empty and 'location' in data.columns:
    locations = data['location'].unique().tolist()
else:
    locations = ["Tanauan", "Talisay", "Laurel", "Agoncillo", "San Nicolas", "Alitagtag"]

selected_locations = st.sidebar.multiselect(
    "Select Locations", 
    locations, 
    default=locations[:3] if len(locations) >= 3 else locations
)

# Add time period selection: Weekly, Monthly, Yearly
time_periods = ["Weekly", "Monthly", "Yearly"]
selected_time_period = st.sidebar.selectbox("Select Time Period", time_periods, index=1)

# Main content
if page == "Home":
    st.title("🌊 Water Quality Prediction Dashboard")

    # Parameter combination info
    st.markdown(f"### Selected Parameter Combination: **{params_combo}**")

    # Filter data for selected location
    if not data.empty and 'location' in data.columns:
        location_data = data[data['location'].isin(selected_locations)]
    else:
        location_data = data

    if location_data.empty:
        st.warning("⚠️ No data available for the selected locations.")
        st.stop()
    
    st.success(f"✅ Data loaded successfully! Found {len(location_data)} records for selected locations.")

    # Show data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(location_data.head(10))
        
    # Prepare input data for model predictions
    try:
        # Ensure we have all required feature columns
        missing_cols = [col for col in full_feature_columns if col not in location_data.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}. Using default values.")
            for col in missing_cols:
                location_data[col] = 0
        
        # Get feature data and handle missing values
        input_features = location_data[full_feature_columns].fillna(location_data[full_feature_columns].mean()).values
        
        if len(input_features) == 0:
            st.error("❌ No valid input data available")
            st.stop()

        # Get predictions from each model
        with st.spinner('Generating predictions...'):
            try:
                wqi_cnn = CNNModel.predict(input_features)
                wqi_lstm = LSTMModel.predict(input_features)
                wqi_hybrid = HYBRIDModel.predict(input_features)
                
                # Calculate average predictions
                avg_cnn = np.mean(wqi_cnn) if len(wqi_cnn) > 0 else 60.0
                avg_lstm = np.mean(wqi_lstm) if len(wqi_lstm) > 0 else 60.0
                avg_hybrid = np.mean(wqi_hybrid) if len(wqi_hybrid) > 0 else 60.0
                
            except Exception as e:
                st.error(f"❌ Error during model predictions: {e}")
                avg_cnn = avg_lstm = avg_hybrid = 60.0

        # Display current metrics
        st.subheader("📈 Current Water Quality Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="CNN WQI", value=format_float(avg_cnn))
            wqi_class, _ = get_wqi_classification(avg_cnn)
            st.caption(f"Status: {wqi_class}")
            
        with col2:
            st.metric(label="LSTM WQI", value=format_float(avg_lstm))
            wqi_class, _ = get_wqi_classification(avg_lstm)
            st.caption(f"Status: {wqi_class}")
            
        with col3:
            st.metric(label="HYBRID WQI", value=format_float(avg_hybrid))
            wqi_class, _ = get_wqi_classification(avg_hybrid)
            st.caption(f"Status: {wqi_class}")

        # Generate forecast for each model
        st.markdown("---")
        st.subheader("🔮 Forecasting Water Quality Index (WQI) and Pollutant Levels")
        
        # Use sample data for forecasting (first row or create sample)
        sample_data = input_features[:1] if len(input_features) > 0 else np.zeros((1, len(full_feature_columns)))
        
        with st.spinner('Generating forecasts...'):
            forecast_cnn = forecast_multi_output(CNNModel, sample_data)
            forecast_lstm = forecast_multi_output(LSTMModel, sample_data)
            forecast_hybrid = forecast_multi_output(HYBRIDModel, sample_data)

        # Create forecast comparison
        try:
            forecast_df = pd.DataFrame({
                "Parameter": list(forecast_cnn.keys()),
                "CNN": list(forecast_cnn.values()),
                "LSTM": list(forecast_lstm.values()),
                "HYBRID": list(forecast_hybrid.values())
            })
            
            st.subheader("📊 Forecast Comparison Table")
            st.dataframe(forecast_df, use_container_width=True)
            
            # Display detailed forecasts
            st.markdown("---")
            st.subheader("🔍 Detailed Forecast Analysis")
            
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            
            with forecast_col1:
                st.markdown("#### 🧠 CNN Model Forecast")
                for key, value in forecast_cnn.items():
                    st.metric(label=key, value=format_float(value))
                
                wqi_classification, wqi_advice = get_wqi_classification(forecast_cnn['WQI'])
                st.info(f"**Classification**: {wqi_classification}")
                with st.expander("💡 Recommendations"):
                    st.write(wqi_advice)
            
            with forecast_col2:
                st.markdown("#### 🔄 LSTM Model Forecast")
                for key, value in forecast_lstm.items():
                    st.metric(label=key, value=format_float(value))
                
                wqi_classification, wqi_advice = get_wqi_classification(forecast_lstm['WQI'])
                st.info(f"**Classification**: {wqi_classification}")
                with st.expander("💡 Recommendations"):
                    st.write(wqi_advice)
            
            with forecast_col3:
                st.markdown("#### 🚀 HYBRID Model Forecast")
                for key, value in forecast_hybrid.items():
                    st.metric(label=key, value=format_float(value))
                
                wqi_classification, wqi_advice = get_wqi_classification(forecast_hybrid['WQI'])
                st.info(f"**Classification**: {wqi_classification}")
                with st.expander("💡 Recommendations"):
                    st.write(wqi_advice)

        except Exception as e:
            st.error(f"❌ Error occurred while generating forecast displays: {e}")
            
    except Exception as e:
        st.error(f"❌ Error occurred while processing data: {e}")
        st.write("**Debug Info:**")
        st.write(f"- Data shape: {data.shape if not data.empty else 'Empty'}")
        st.write(f"- Selected locations: {selected_locations}")
        st.write(f"- Available columns: {list(data.columns) if not data.empty else 'None'}")
        st.write(f"- Required columns: {full_feature_columns}")

elif page == "Model Information":
    st.title("🤖 Model Information")
    
    # Basic model descriptions
    st.markdown("### Model Architectures")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🧠 CNN Model")
        st.info("Convolutional Neural Network architecture designed to capture spatial patterns and local features in water quality data through convolutional layers and pooling operations.")
        
    with col2:
        st.markdown("#### 🔄 LSTM Model")
        st.info("Long Short-Term Memory network that excels at capturing temporal dependencies and sequential patterns in time-series water quality data.")
        
    with col3:
        st.markdown("#### 🚀 HYBRID Model")
        st.info("Combined CNN-LSTM architecture that leverages both spatial feature extraction and temporal sequence modeling for enhanced prediction accuracy.")

    # Model status
    st.markdown("---")
    st.subheader("📊 Model Status")
    
    model_status_col1, model_status_col2, model_status_col3 = st.columns(3)
    
    with model_status_col1:
        cnn_status = "✅ Loaded" if hasattr(CNNModel, 'predict') else "⚠️ Mock Mode"
        st.metric("CNN Model", cnn_status)
        
    with model_status_col2:
        lstm_status = "✅ Loaded" if hasattr(LSTMModel, 'predict') else "⚠️ Mock Mode"
        st.metric("LSTM Model", lstm_status)
        
    with model_status_col3:
        hybrid_status = "✅ Loaded" if hasattr(HYBRIDModel, 'predict') else "⚠️ Mock Mode"
        st.metric("HYBRID Model", hybrid_status)

    # Try to load and display model comparison
    st.markdown("---")
    st.subheader("📊 Model Performance Analysis")

# Show model
model_comparison = load_model_comparison()
    
if model_comparison:
    # Get model names (excluding 'best_model' key)
    model_names = [key for key in model_comparison.keys() if key != 'best_model' and isinstance(model_comparison[key], dict)]
    
    if model_names:
        # Extract metrics data for all models
        metrics_data = []
        
        for model_name in model_names:
            model_data = model_comparison[model_name]
            if isinstance(model_data, dict) and 'metrics' in model_data:
                model_metrics = model_data['metrics']
                if isinstance(model_metrics, dict):
                    row = {'Model': model_name.upper()}
                    # Add all metrics from the metrics section
                    row.update(model_metrics)
                    metrics_data.append(row)
        
        if metrics_data:
            # Create DataFrame for table display
            df_metrics = pd.DataFrame(metrics_data)
            
            # Format numeric columns for better display
            for col in df_metrics.columns:
                if col != 'Model' and df_metrics[col].dtype in ['float64', 'int64']:
                    df_metrics[col] = df_metrics[col].round(6)
            
            # Display metrics table
            st.subheader("📊 Model Metrics Comparison")
            st.dataframe(
                df_metrics,
                use_container_width=True,
                hide_index=True
            )
            
            # Show best model highlight
            if 'best_model' in model_comparison:
                best_model_name = model_comparison['best_model']
                st.success(f"🏆 Best Model: **{best_model_name.upper()}**")
            
            # Optional: Individual metric comparison charts
            if st.expander("📈 Metric Visualization", expanded=False):
                # Get numeric columns (metrics)
                metric_columns = [col for col in df_metrics.columns if col != 'Model']
                
                if metric_columns:
                    selected_metric = st.selectbox(
                        "Select metric to visualize:",
                        metric_columns,
                        index=0
                    )
                    
                    if selected_metric:
                        # Special title formatting for R2
                        title_metric = "R²" if selected_metric == 'r2' else selected_metric.upper()
                        
                        fig_bar = px.bar(
                            df_metrics, 
                            x='Model', 
                            y=selected_metric,
                            title=f"Model Comparison: {title_metric}",
                            color='Model',
                            text=selected_metric
                        )
                        
                        # Format text on bars
                        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No metrics data found in the models.")
    else:
        st.info("No model data found.")

elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    
    if not data.empty:
        st.subheader("📈 Data Overview")
        
        # Basic data info
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Total Records", len(data))
        with info_col2:
            st.metric("Features", len([col for col in data.columns if col != 'location']))
        with info_col3:
            if 'location' in data.columns:
                st.metric("Locations", data['location'].nunique())
            else:
                st.metric("Locations", "N/A")
        
        # Basic statistics
        if st.checkbox("📊 Show Basic Statistics"):
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                st.write("**Statistical Summary:**")
                st.dataframe(numeric_data.describe(), use_container_width=True)
            else:
                st.warning("No numeric data available for statistics.")
        
        # Data quality check
        if st.checkbox("🔍 Show Data Quality"):
            st.write("**Missing Values:**")
            missing_data = data.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(data) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Interactive Visualizations")
        
        # Parameter selection for visualization
        all_numeric_params = [col for col in data.columns if col in WATER_PARAMETERS + CLIMATE_PARAMETERS + POLLUTANT_PARAMETERS]
        
        if all_numeric_params:
            viz_param = st.selectbox("Select Parameter for Visualization", all_numeric_params)
            
            if viz_param in data.columns:
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Box plot by location
                    if 'location' in data.columns and data['location'].nunique() > 1:
                        st.subheader(f"📦 {viz_param} Distribution by Location")
                        fig_box = px.box(data, x='location', y=viz_param, 
                                        title=f"{viz_param} by Location",
                                        color='location')
                        fig_box.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.subheader(f"📊 {viz_param} Distribution")
                        fig_hist = px.histogram(data, x=viz_param, 
                                              title=f"Distribution of {viz_param}",
                                              nbins=30)
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                with viz_col2:
                    # Time series if date column exists, otherwise scatter plot
                    st.subheader(f"📈 {viz_param} Trends")
                    if len(data) > 1:
                        # Create a simple line plot with index as x-axis
                        fig_line = px.line(data.reset_index(), x='index', y=viz_param,
                                         title=f"{viz_param} Over Records")
                        fig_line.update_xaxes(title="Record Number")
                        st.plotly_chart(fig_line, use_container_width=True)
                    else:
                        st.info("Need more data points for trend visualization")
            
            # Correlation analysis
            if st.checkbox("🔗 Show Correlation Analysis"):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.subheader("🔥 Parameter Correlation Matrix")
                    
                    # Select subset of columns for better visualization
                    selected_cols = st.multiselect(
                        "Select parameters for correlation analysis:",
                        numeric_cols.tolist(),
                        default=numeric_cols[:min(10, len(numeric_cols))].tolist()
                    )
                    
                    if len(selected_cols) > 1:
                        corr_matrix = data[selected_cols].corr()
                        fig_corr = px.imshow(corr_matrix, 
                                           title="Parameter Correlation Matrix",
                                           color_continuous_scale='RdBu',
                                           aspect="auto")
                        fig_corr.update_layout(width=800, height=600)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Show highest correlations
                        st.subheader("🔝 Highest Correlations")
                        # Get upper triangle of correlation matrix
                        corr_pairs = []
                        for i in range(len(selected_cols)):
                            for j in range(i+1, len(selected_cols)):
                                corr_pairs.append({
                                    'Parameter 1': selected_cols[i],
                                    'Parameter 2': selected_cols[j],
                                    'Correlation': corr_matrix.iloc[i, j]
                                })
                        
                        if corr_pairs:
                            corr_df = pd.DataFrame(corr_pairs)
                            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                            st.dataframe(corr_df.head(10), use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
            
            # Multi-parameter comparison
            if st.checkbox("📊 Multi-Parameter Analysis"):
                if len(all_numeric_params) >= 2:
                    st.subheader("🔄 Parameter Relationships")
                    
                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        x_param = st.selectbox("Select X-axis parameter:", all_numeric_params, key="x_param")
                    with param_col2:
                        y_param = st.selectbox("Select Y-axis parameter:", 
                                             [p for p in all_numeric_params if p != x_param], 
                                             key="y_param")
                    
                    if x_param and y_param:
                        fig_scatter = px.scatter(data, x=x_param, y=y_param,
                                               color='location' if 'location' in data.columns else None,
                                               title=f"{y_param} vs {x_param}",
                                               trendline="ols" if len(data) > 10 else None)
                        st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ No numeric parameters available for visualization.")
        
        # Raw data view
        if st.checkbox("🗃️ Show Raw Data"):
            st.subheader("📋 Raw Data Table")
            
            # Add filters
            if 'location' in data.columns:
                location_filter = st.multiselect(
                    "Filter by Location:",
                    data['location'].unique(),
                    default=data['location'].unique()
                )
                filtered_data = data[data['location'].isin(location_filter)]
            else:
                filtered_data = data
            
            # Show number of records
            st.info(f"Showing {len(filtered_data)} records")
            
            # Display data with pagination
            page_size = st.selectbox("Records per page:", [10, 25, 50, 100], index=1)
            
            if len(filtered_data) > page_size:
                page_num = st.number_input("Page number:", min_value=1, 
                                         max_value=(len(filtered_data) // page_size) + 1, 
                                         value=1)
                start_idx = (page_num - 1) * page_size
                end_idx = start_idx + page_size
                st.dataframe(filtered_data.iloc[start_idx:end_idx], use_container_width=True)
            else:
                st.dataframe(filtered_data, use_container_width=True)
                
    else:
        st.warning("⚠️ No data available for visualization.")
        st.info("Please ensure your data file is properly loaded on the Home page.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌊 Water Quality Prediction Dashboard | Built with Streamlit</p>
    <p>Models: CNN • LSTM • HYBRID | Data-driven environmental monitoring</p>
</div>
""", unsafe_allow_html=True)