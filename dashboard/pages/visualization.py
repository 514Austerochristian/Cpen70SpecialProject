# dashboard/pages/visualization.py
"""
Data Visualization page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config.parameters import WATER_PARAMETERS, CLIMATE_PARAMETERS, POLLUTANT_PARAMETERS

def render(data):
    """Render the data visualization page"""
    st.title("📊 Data Visualization")
    
    if data.empty:
        st.warning("⚠️ No data available for visualization.")
        st.info("Please ensure your data file is properly loaded on the Home page.")
        return
    
    st.subheader("📈 Data Overview")
    
    # Metrics
    location_col1, location_col2, location_col3, location_col4 = st.columns(4)
    with location_col1:
        st.metric("Total Records", len(data))
    with location_col2:
        st.metric("Features", len([col for col in data.columns if col != 'location']))
    with location_col3:
        if 'location' in data.columns:
            st.metric("Total Locations", data['location'].nunique())
        else:
            st.metric("Locations", "N/A")
    with location_col4:
        if 'location' in data.columns:
            avg_records = len(data) / data['location'].nunique()
            st.metric("Avg Records/Location", f"{avg_records:.1f}")
    
    # Location analysis
    if 'location' in data.columns:
        with st.expander("🔍 Location Analysis", expanded=False):
            st.subheader("Records per Location")
            location_counts = data['location'].value_counts().sort_values(ascending=False)
            
            fig_locations = px.bar(
                x=location_counts.index,
                y=location_counts.values,
                title="Data Records by Location",
                labels={'x': 'Location', 'y': 'Number of Records'}
            )
            fig_locations.update_xaxes(tickangle=45)
            st.plotly_chart(fig_locations, use_container_width=True)
            
            st.dataframe(
                location_counts.to_frame('Record Count').reset_index().rename(columns={'index': 'Location'}),
                use_container_width=True
            )
    
    # Statistics
    if st.checkbox("📊 Show Basic Statistics"):
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.write("**Statistical Summary:**")
            st.dataframe(numeric_data.describe(), use_container_width=True)
        else:
            st.warning("No numeric data available for statistics.")
    
    # Data quality
    if st.checkbox("🔍 Show Data Quality"):
        st.write("**Missing Values:**")
        missing_data = data.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(data) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Interactive Visualizations")
    
    all_numeric_params = [col for col in data.columns if col in WATER_PARAMETERS + CLIMATE_PARAMETERS + POLLUTANT_PARAMETERS]
    
    if all_numeric_params:
        viz_param = st.selectbox("Select Parameter for Visualization", all_numeric_params)
        
        if viz_param in data.columns:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
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
                st.subheader(f"📈 {viz_param} Trends")
                if len(data) > 1:
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
                    
                    st.subheader("🔍 Highest Correlations")
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
    
    # Raw data
    if st.checkbox("🗃️ Show Raw Data"):
        st.subheader("📋 Raw Data Table")
        
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            if 'location' in data.columns:
                location_filter = st.multiselect(
                    "Filter by Location:",
                    sorted(data['location'].unique()),
                    default=sorted(data['location'].unique())[:5] if len(data['location'].unique()) > 5 else sorted(data['location'].unique())
                )
                if location_filter:
                    filtered_data = data[data['location'].isin(location_filter)]
                else:
                    filtered_data = pd.DataFrame()
            else:
                filtered_data = data
                st.info("No location column available for filtering")
        
        with col_filter2:
            if 'location' in data.columns:
                st.info(f"🔍 {len(location_filter) if 'location_filter' in locals() else 0} locations selected")
        
        if not filtered_data.empty:
            st.success(f"✅ Showing {len(filtered_data)} records")
            
            page_size = st.selectbox("Records per page:", [10, 25, 50, 100], index=1)
            
            if len(filtered_data) > page_size:
                max_pages = (len(filtered_data) - 1) // page_size + 1
                page_num = st.number_input("Page number:", min_value=1, 
                                         max_value=max_pages, 
                                         value=1)
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, len(filtered_data))
                st.info(f"Showing records {start_idx + 1} to {end_idx} of {len(filtered_data)}")
                st.dataframe(filtered_data.iloc[start_idx:end_idx], use_container_width=True)
            else:
                st.dataframe(filtered_data, use_container_width=True)
        else:
            st.warning("⚠️ No data available with current filters. Please select at least one location.")