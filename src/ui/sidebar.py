"""
Sidebar UI component
"""

import streamlit as st
from typing import Optional


def create_sidebar() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """
    Create sidebar with file upload and information
    
    Returns:
        Uploaded file object or None
    """
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your gym sensor data in CSV format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Loaded: {uploaded_file.name}")
            
            # File information
            file_size = uploaded_file.size / (1024 * 1024)
            st.info(f"File size: {file_size:.2f} MB")
            
            # Additional information
            st.markdown("---")
            st.subheader("📊 Analysis Options")
            
            # Analysis settings could go here
            st.markdown("""
            **Available Analyses:**
            - Data Overview & Validation
            - Outlier Detection
            - Timing Analysis
            - Feature Extraction
            - Machine Learning
            - Signal Visualization
            - Anomaly Investigation
            """)
        else:
            # Instructions when no file is uploaded
            st.info("Please upload a CSV file to begin")
            
            st.markdown("---")
            st.subheader("📋 Required Format")
            st.markdown("""
            **Sensor Columns:**
            - `ax`, `ay`, `az` (accelerometer)
            - `gx`, `gy`, `gz` (gyroscope)
            
            **Metadata Columns:**
            - `athlete_id`
            - `exercise_type`
            - `weight_kg`
            - `set_number`
            - `rep_number`
            
            **Optional:**
            - `timestamp` (recommended)
            """)
            
            st.markdown("---")
            st.subheader("🔧 Settings")
            
            # Global settings
            if st.checkbox("Show advanced options"):
                st.number_input(
                    "Outlier Z-score threshold",
                    min_value=2.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    key="outlier_threshold"
                )
                
                st.number_input(
                    "Sampling rate (Hz)",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    key="sampling_rate"
                )
        
        # About section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Gym Movement Sensor Analysis**
        
        Version: 1.0.0
        
        This dashboard provides comprehensive analysis of gym movement sensor data including:
        - Statistical analysis
        - Signal processing
        - Machine learning
        - Anomaly detection
        """)
        
        # Resources
        with st.expander("📚 Resources"):
            st.markdown("""
            - [Documentation](#)
            - [Sample Data](#)
            - [Tutorial](#)
            - [GitHub](#)
            """)
    
    return uploaded_file


def create_sidebar_with_filters(df) -> dict:
    """
    Create sidebar with data filters
    
    Args:
        df: DataFrame to filter
        
    Returns:
        Dictionary of filter selections
    """
    filters = {}
    
    with st.sidebar:
        st.subheader("🔍 Data Filters")
        
        # Exercise type filter
        if 'exercise_type' in df.columns:
            exercises = ['All'] + sorted(df['exercise_type'].unique().tolist())
            filters['exercise'] = st.selectbox("Exercise Type", exercises)
        
        # Athlete filter
        if 'athlete_id' in df.columns:
            athletes = ['All'] + sorted(df['athlete_id'].unique().tolist())
            filters['athlete'] = st.selectbox("Athlete", athletes)
        
        # Weight range filter
        if 'weight_kg' in df.columns:
            weight_min = float(df['weight_kg'].min())
            weight_max = float(df['weight_kg'].max())
            
            filters['weight_range'] = st.slider(
                "Weight Range (kg)",
                min_value=weight_min,
                max_value=weight_max,
                value=(weight_min, weight_max)
            )
        
        # Set number filter
        if 'set_number' in df.columns:
            sets = ['All'] + sorted(df['set_number'].unique().tolist())
            filters['set'] = st.multiselect("Set Numbers", sets, default=['All'])
        
        # Apply filters button
        filters['apply'] = st.button("Apply Filters", type="primary")
        
        # Reset filters button
        if st.button("Reset Filters"):
            st.experimental_rerun()
    
    return filters


def apply_filters(df, filters: dict):
    """
    Apply filters to dataframe
    
    Args:
        df: Original DataFrame
        filters: Dictionary of filter selections
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply exercise filter
    if filters.get('exercise') and filters['exercise'] != 'All':
        filtered_df = filtered_df[filtered_df['exercise_type'] == filters['exercise']]
    
    # Apply athlete filter
    if filters.get('athlete') and filters['athlete'] != 'All':
        filtered_df = filtered_df[filtered_df['athlete_id'] == filters['athlete']]
    
    # Apply weight range filter
    if filters.get('weight_range'):
        min_weight, max_weight = filters['weight_range']
        filtered_df = filtered_df[
            (filtered_df['weight_kg'] >= min_weight) & 
            (filtered_df['weight_kg'] <= max_weight)
        ]
    
    # Apply set filter
    if filters.get('set') and 'All' not in filters['set']:
        filtered_df = filtered_df[filtered_df['set_number'].isin(filters['set'])]
    
    return filtered_df