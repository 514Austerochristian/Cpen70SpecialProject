# dashboard/models/loader.py
"""
Model loading functions
"""

import os
import tensorflow as tf
import streamlit as st
from utils.paths import CNN_MODEL_PATH, LSTM_MODEL_PATH, HYBRID_MODEL_PATH
import numpy as np

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
            if os.path.exists(CNN_MODEL_PATH):
                return tf.keras.models.load_model(CNN_MODEL_PATH, custom_objects=custom_objects)
        elif model_name == "LSTM":
            if os.path.exists(LSTM_MODEL_PATH):
                return tf.keras.models.load_model(LSTM_MODEL_PATH, custom_objects=custom_objects)
        elif model_name == "HYBRID":
            if os.path.exists(HYBRID_MODEL_PATH):
                return tf.keras.models.load_model(HYBRID_MODEL_PATH, custom_objects=custom_objects)
        
        st.warning(f"⚠️ {model_name} model file not found. Using mock predictions.")
        return create_mock_model(model_name)
        
    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {e}")
        return create_mock_model(model_name)

def create_mock_model(model_name):
    """Create a mock model for demonstration"""
    class MockModel:
        def __init__(self, name):
            self.name = name
            
        def predict(self, X):
            n_samples = len(X) if hasattr(X, '__len__') else 1
            if n_samples > 0:
                if self.name == "CNN":
                    base_wqi = np.random.normal(58, 8, n_samples)
                elif self.name == "LSTM":
                    base_wqi = np.random.normal(62, 7, n_samples)
                else:
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

def load_all_models():
    """Load all three models"""
    with st.spinner('Loading models...'):
        return {
            'CNN': load_model("CNN"),
            'LSTM': load_model("LSTM"),
            'HYBRID': load_model("HYBRID")
        }
    