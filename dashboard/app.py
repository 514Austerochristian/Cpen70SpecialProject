# dashboard/app.py
"""
Main Streamlit Application Entry Point
Handles navigation and page routing only
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
import streamlit as st
from config.settings import configure_page
from ui.sidebar import render_sidebar
from pages import home, model_info, visualization
from data.loader import load_and_preprocess_data
from models.loader import load_all_models

# Configure page (MUST BE FIRST)
configure_page()

# Load shared resources
@st.cache_data
def get_data():
    return load_and_preprocess_data()

@st.cache_resource
def get_models():
    return load_all_models()

# Load data and models
data = get_data()
models = get_models()

# Render sidebar and get selections
page, sidebar_state = render_sidebar(data)

# Page routing
if page == "Home":
    home.render(data, models, sidebar_state)
elif page == "Model Information":
    model_info.render(models)
elif page == "Data Visualization":
    visualization.render(data)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌊 Water Quality Prediction Dashboard | Built with Streamlit</p>
    <p>Models: CNN • LSTM • HYBRID | Data-driven environmental monitoring</p>
    <p>📍 Monitoring {sidebar_state.get('total_locations', 0)} locations | 🔬 Analyzing 14 parameters</p>
</div>
""", unsafe_allow_html=True)