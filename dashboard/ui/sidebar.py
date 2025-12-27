"""
Sidebar components
"""

import streamlit as st
from config.parameters import PARAMETER_COMBINATIONS
from config.settings import TIME_PERIODS
from data.utils import get_all_unique_locations

def render_sidebar(data):
    """Render sidebar and return page selection and state"""
    st.sidebar.title("Water Quality Forecasting")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox("Select Page", ["Home", "Model Information", "Data Visualization"])
    
    # Parameter selection
    st.sidebar.subheader("Select Parameters")
    params_combo = st.sidebar.selectbox(
        "Select Parameter Combination", 
        list(PARAMETER_COMBINATIONS.keys()), 
        help="Choose a parameter combination for forecasting."
    )
    
    # Location filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Locations")
    
    all_locations = get_all_unique_locations(data)
    
    st.sidebar.info(f"📍 Total locations available: {len(all_locations)}")
    
    selected_locations = st.sidebar.multiselect(
        "Select Locations", 
        all_locations, 
        default=all_locations[:5] if len(all_locations) >= 5 else all_locations,
        help="Select one or more locations to analyze."
    )
    
    if st.sidebar.button("🌍 Select All Locations"):
        selected_locations = all_locations
    
    if selected_locations:
        st.sidebar.success(f"✅ {len(selected_locations)} location(s) selected")
    else:
        st.sidebar.warning("⚠️ No locations selected")
    
    # Time period selection
    selected_time_period = st.sidebar.selectbox("Select Time Period", TIME_PERIODS, index=1)
    
    # Return page and sidebar state
    sidebar_state = {
        'params_combo': params_combo,
        'selected_locations': selected_locations,
        'all_locations': all_locations,
        'time_period': selected_time_period,
        'total_locations': len(all_locations)
    }
    
    return page, sidebar_state