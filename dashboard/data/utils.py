# dashboard/data/utils.py
"""
Data utility functions
"""

import pandas as pd
import streamlit as st

def get_all_unique_locations(data):
    """
    Extract all unique locations from the loaded data
    
    Args:
        data: Pandas DataFrame containing the loaded data
    
    Returns:
        list: All unique location names from the data
    
    Raises:
        Exception: If no valid locations are found in the data
    """
    # Check if data exists
    if data is None or data.empty:
        st.error("❌ **CRITICAL ERROR**: No data loaded from CSV file!")
        st.error("Please ensure 'model_processed_data.csv' exists in the correct location.")
        st.info("Expected file locations:")
        st.code("""
        - /workspaces/Cpen70SpecialProject/data/processed/model_processed_data.csv
        - /workspaces/Cpen70SpecialProject/data/model_processed_data.csv
        - data/processed/model_processed_data.csv
        - data/model_processed_data.csv
        """)
        st.stop()  # Stop execution immediately
    
    # Check if 'location' column exists
    if 'location' not in data.columns and 'Location' not in data.columns:
        st.error("❌ **CRITICAL ERROR**: No 'location' or 'Location' column found in the CSV file!")
        st.error(f"Available columns in your CSV: {list(data.columns)}")
        st.info("💡 **Solution**: Ensure your CSV file has a column named 'location' or 'Location' with location names.")
        st.stop()  # Stop execution immediately
    
    # Standardize column name
    location_col = 'location' if 'location' in data.columns else 'Location'
    
    # Extract unique locations
    unique_locations = sorted(data[location_col].dropna().unique().tolist())
    
    # Check if we have valid locations
    if not unique_locations or len(unique_locations) == 0:
        st.error("❌ **CRITICAL ERROR**: No valid locations found in the 'location' column!")
        st.error("The 'location' column exists but contains no valid data.")
        st.info(f"Total rows in CSV: {len(data)}")
        st.info(f"Rows with missing location: {data[location_col].isna().sum()}")
        st.stop()  # Stop execution immediately
    
    # Success - return locations
    st.success(f"✅ Successfully loaded {len(unique_locations)} locations from CSV file")
    return unique_locations

def format_float(value):
    """Safely format float values"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.2f}"