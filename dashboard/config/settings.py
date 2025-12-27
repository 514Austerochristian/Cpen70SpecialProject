# dashboard/config/settings.py
"""
Application configuration and settings
"""

import streamlit as st

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Comprehensive Model Water Quality Prediction Comparison Dashboard",
        page_icon="🌊",
        layout="wide"
    )
    
    # Apply custom CSS
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

# Time periods
TIME_PERIODS = ["Weekly", "Monthly", "Yearly"]