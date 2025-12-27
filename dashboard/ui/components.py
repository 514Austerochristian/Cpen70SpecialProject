"""
Reusable UI components
"""

import streamlit as st

def create_metric_card(wqi_classification, wqi_advice):
    """Create a metric card for WQI insights"""
    st.subheader("Water Quality Index (WQI) Insights")
    st.metric(label="WQI Classification", value=wqi_classification)
    st.write(wqi_advice)