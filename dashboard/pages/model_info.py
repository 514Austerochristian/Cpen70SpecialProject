# dashboard/pages/model_info.py
"""
Model Information page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from models.comparison import load_model_comparison

def render(models):
    """Render the model information page"""
    st.title("🤖 Model Information")
    
    # Model architectures
    st.markdown("### Model Architectures")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🧠 CNN Model")
        st.info("Convolutional Neural Network architecture designed to capture spatial patterns and local features in water quality data.")
    
    with col2:
        st.markdown("#### 🔄 LSTM Model")
        st.info("Long Short-Term Memory network that excels at capturing temporal dependencies and sequential patterns in time-series water quality data.")
    
    with col3:
        st.markdown("#### 🚀 HYBRID Model")
        st.info("Combined CNN-LSTM architecture that leverages both spatial feature extraction and temporal sequence modeling.")
    
    # Multi-output system
    st.markdown("---")
    st.subheader("🔄 Multi-Output Prediction System")
    st.markdown("This dashboard leverages a multi-output prediction system for simultaneous forecasts of various water quality parameters.")
    
    # Model status
    st.markdown("---")
    st.subheader("📊 Model Status")
    
    model_status_col1, model_status_col2,model_status_col3 = st.columns(3)

    with model_status_col1:
        cnn_status = "✅ Loaded" if hasattr(models['CNN'], 'predict') else "⚠️ Mock Mode"
        st.metric("CNN Model", cnn_status)
    with model_status_col2:
        lstm_status = "✅ Loaded" if hasattr(models['LSTM'], 'predict') else "⚠️ Mock Mode"
        st.metric("LSTM Model", lstm_status)

    with model_status_col3:
        hybrid_status = "✅ Loaded" if hasattr(models['HYBRID'], 'predict') else "⚠️ Mock Mode"
        st.metric("HYBRID Model", hybrid_status)

# Model comparison
st.markdown("---")
st.subheader("📊 Model Performance Analysis")

model_comparison = load_model_comparison()

if model_comparison:
    model_names = [key for key in model_comparison.keys() if key != 'best_model' and isinstance(model_comparison[key], dict)]
    
    if model_names:
        metrics_data = []
        
        for model_name in model_names:
            model_data = model_comparison[model_name]
            if isinstance(model_data, dict) and 'metrics' in model_data:
                model_metrics = model_data['metrics']
                if isinstance(model_metrics, dict):
                    row = {'Model': model_name.upper()}
                    row.update(model_metrics)
                    metrics_data.append(row)
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            for col in df_metrics.columns:
                if col != 'Model' and df_metrics[col].dtype in ['float64', 'int64']:
                    df_metrics[col] = df_metrics[col].round(6)
            
            st.subheader("📊 Model Metrics Comparison")
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            if 'best_model' in model_comparison:
                best_model_name = model_comparison['best_model']
                st.success(f"🏆 Best Model: **{best_model_name.upper()}**")
            
            if st.expander("📈 Metric Visualization", expanded=False):
                metric_columns = [col for col in df_metrics.columns if col != 'Model']
                
                if metric_columns:
                    selected_metric = st.selectbox("Select metric to visualize:", metric_columns, index=0)
                    
                    if selected_metric:
                        title_metric = "R²" if selected_metric == 'r2' else selected_metric.upper()
                        
                        fig_bar = px.bar(
                            df_metrics, 
                            x='Model', 
                            y=selected_metric,
                            title=f"Model Comparison: {title_metric}",
                            color='Model',
                            text=selected_metric
                        )
                        
                        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)