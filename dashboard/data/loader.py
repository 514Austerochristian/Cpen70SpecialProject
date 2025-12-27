# dashboard/data/loader.py
"""
Data loading and preprocessing functions
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from utils.paths import BASE_DIR
from config.parameters import FULL_FEATURE_COLUMNS

def load_and_preprocess_data():
    """Load and preprocess data for different parameter combinations"""
    
    # Try multiple possible paths for the data file
    possible_paths = [
        os.path.join(BASE_DIR, '..', 'data', 'processed', 'model_processed_data.csv'),
        os.path.join(BASE_DIR, '..', 'data', 'model_processed_data.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'model_processed_data.csv'),
        '/workspaces/Cpen70SpecialProject/data/processed/model_processed_data.csv',
        '/workspaces/Cpen70SpecialProject/data/model_processed_data.csv',
        'data/processed/model_processed_data.csv',
        'data/model_processed_data.csv'
    ]
    
    data = None
    found_path = None
    
    # Try each path
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            try:
                data = pd.read_csv(normalized_path)
                found_path = normalized_path
                st.success(f"✅ Data file found: {normalized_path}")
                break
            except Exception as e:
                st.warning(f"⚠️ Found file at {normalized_path} but couldn't read it: {e}")
                continue
    
    # If no file found, show error and stop
    if data is None:
        st.error("❌ **CRITICAL ERROR**: Could not find 'model_processed_data.csv' file!")
        st.error("**Searched in the following locations:**")
        for path in possible_paths:
            st.code(os.path.normpath(path))
        
        st.info("💡 **Solution**: Please ensure your CSV file is placed in one of the locations above.")
        st.info("🔍 **Current working directory**: " + os.getcwd())
        
        # Show what files exist in data directories
        data_dirs = [
            os.path.join(BASE_DIR, '..', 'data'),
            os.path.join(BASE_DIR, '..', 'data', 'processed'),
            '/workspaces/Cpen70SpecialProject/data',
            '/workspaces/Cpen70SpecialProject/data/processed'
        ]
        
        st.write("**Available data directories and their contents:**")
        for data_dir in data_dirs:
            norm_dir = os.path.normpath(data_dir)
            if os.path.exists(norm_dir):
                try:
                    files = os.listdir(norm_dir)
                    csv_files = [f for f in files if f.endswith('.csv')]
                    st.write(f"📁 {norm_dir}:")
                    if csv_files:
                        for f in csv_files:
                            st.write(f"   - {f}")
                    else:
                        st.write("   (no CSV files found)")
                except Exception as e:
                    st.write(f"   Error reading directory: {e}")
        
        st.stop()  # Stop execution immediately
    
    # Validate data is not empty
    if data.empty:
        st.error("❌ **CRITICAL ERROR**: CSV file is empty!")
        st.error(f"File location: {found_path}")
        st.stop()
    
    st.info(f"📊 Loaded {len(data)} rows and {len(data.columns)} columns from CSV")
    
    # Check for location column with case-insensitive search
    location_col = None
    for col in data.columns:
        if col.lower() == 'location':
            location_col = col
            break
    
    if location_col:
        # Standardize to lowercase 'location'
        if location_col != 'location':
            data.rename(columns={location_col: 'location'}, inplace=True)
        st.info(f"📍 Found location column with {data['location'].nunique()} unique locations")
    else:
        st.error("❌ **CRITICAL ERROR**: No 'location' column found in CSV!")
        st.error(f"Available columns: {list(data.columns)}")
        st.info("💡 **Solution**: Add a 'location' column to your CSV file with location names.")
        st.stop()
    
    # Ensure all required columns exist or can be filled
    missing_cols = []
    for col in FULL_FEATURE_COLUMNS:
        if col not in data.columns:
            missing_cols.append(col)
    
    if missing_cols:
        st.warning(f"⚠️ Missing {len(missing_cols)} expected columns from CSV:")
        for col in missing_cols[:5]:  # Show first 5
            st.write(f"   - {col}")
        if len(missing_cols) > 5:
            st.write(f"   ... and {len(missing_cols) - 5} more")
        
        st.error("**The app requires these specific columns to work properly.**")
        st.info("💡 **Solution**: Ensure your CSV has all required water quality and climate parameters.")
        st.stop()
    
    st.success("✅ All required columns found in CSV file!")
    return data