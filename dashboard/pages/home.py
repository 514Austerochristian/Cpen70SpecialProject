# dashboard/pages/home.py
"""
Home page - Main dashboard with predictions and forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config.parameters import FULL_FEATURE_COLUMNS
from models.prediction import safe_model_predict, forecast_multi_output_enhanced
from data.utils import format_float
from utils.helpers import get_wqi_classification

def render(data, models, sidebar_state):
    """Render the home page"""
    st.title("🌊 Water Quality Prediction Dashboard")
    
    params_combo = sidebar_state['params_combo']
    selected_locations = sidebar_state['selected_locations']
    all_locations = sidebar_state['all_locations']
    
    st.markdown(f"### Selected Parameter Combination: **{params_combo}**")
    
    # Location information
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Total Available Locations", len(all_locations))
    with col_info2:
        st.metric("Selected Locations", len(selected_locations))
    with col_info3:
        if not data.empty:
            st.metric("Total Records", len(data))
    
    # Show all available locations
    with st.expander("🔍 View All Available Locations", expanded=False):
        st.write("**All locations found in your data:**")
        cols_per_row = 4
        for i in range(0, len(all_locations), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, loc in enumerate(all_locations[i:i+cols_per_row]):
                with cols[j]:
                    is_selected = loc in selected_locations
                    status = "✅" if is_selected else "⭕"
                    st.write(f"{status} {loc}")
    
    # Filter data
    if not selected_locations:
        st.error("❌ Please select at least one location to continue.")
        st.stop()
    
    if not data.empty and 'location' in data.columns:
        location_data = data[data['location'].isin(selected_locations)]
    else:
        location_data = data
    
    if location_data.empty:
        st.warning("⚠️ No data available for the selected locations.")
        st.stop()
    
    # Location distribution
    if 'location' in location_data.columns and len(selected_locations) > 1:
        with st.expander("📊 Location Data Distribution"):
            location_counts = location_data['location'].value_counts()
            st.bar_chart(location_counts)
            st.dataframe(
                location_counts.to_frame('Record Count').reset_index().rename(columns={'index': 'Location'}),
                use_container_width=True
            )
    
    # Data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(location_data.head(10))
    
    # Prepare input data
    try:
        missing_cols = [col for col in FULL_FEATURE_COLUMNS if col not in location_data.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}. Using default values.")
            for col in missing_cols:
                location_data[col] = 0
        
        input_features = location_data[FULL_FEATURE_COLUMNS].fillna(location_data[FULL_FEATURE_COLUMNS].mean()).values
        
        if len(input_features) == 0:
            st.error("❌ No valid input data available")
            st.stop()
        
        # Generate predictions
        with st.spinner('Generating predictions...'):
            try:
                wqi_cnn = safe_model_predict(models['CNN'], input_features, "CNN")
                wqi_lstm = safe_model_predict(models['LSTM'], input_features, "LSTM")
                wqi_hybrid = safe_model_predict(models['HYBRID'], input_features, "HYBRID")
                
                avg_cnn = np.mean(wqi_cnn) if wqi_cnn is not None and len(wqi_cnn) > 0 else 60.0
                avg_lstm = np.mean(wqi_lstm) if wqi_lstm is not None and len(wqi_lstm) > 0 else 60.0
                avg_hybrid = np.mean(wqi_hybrid) if wqi_hybrid is not None and len(wqi_hybrid) > 0 else 60.0
                
                avg_cnn = max(0, min(100, avg_cnn))
                avg_lstm = max(0, min(100, avg_lstm))
                avg_hybrid = max(0, min(100, avg_hybrid))
                
                st.success("✅ Predictions generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during model predictions: {e}")
                avg_cnn = avg_lstm = avg_hybrid = 60.0
        
        # Display metrics
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
        
        # Forecasting
        st.markdown("---")
        st.subheader("🔮 Forecasting Water Quality Index (WQI) and Pollutant Levels")
        
        sample_data = input_features[:1] if len(input_features) > 0 else np.zeros((1, len(FULL_FEATURE_COLUMNS)))
        
        with st.spinner('Generating forecasts...'):
            forecast_cnn = forecast_multi_output_enhanced(models['CNN'], sample_data, "CNN")
            forecast_lstm = forecast_multi_output_enhanced(models['LSTM'], sample_data, "LSTM")
            forecast_hybrid = forecast_multi_output_enhanced(models['HYBRID'], sample_data, "HYBRID")
        
        # Forecast comparison
        try:
            forecast_df = pd.DataFrame({
                "Parameter": list(forecast_cnn.keys()),
                "CNN": list(forecast_cnn.values()),
                "LSTM": list(forecast_lstm.values()),
                "HYBRID": list(forecast_hybrid.values())
            })
            
            st.subheader("📊 Forecast Comparison Table")
            st.dataframe(forecast_df, use_container_width=True)
            
            st.subheader("📈 Forecast Visualization")
            fig = px.bar(forecast_df, x='Parameter', y=['CNN', 'LSTM', 'HYBRID'], barmode='group',
                        title='Forecast Comparison by Model')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed forecasts
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
        