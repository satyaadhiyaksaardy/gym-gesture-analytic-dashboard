"""
Main Streamlit application for Gym Movement Sensor Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime
import gc
import warnings

# Import configuration
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE,
    CUSTOM_CSS, TABS
)

# Import modules
from data.loader import DataLoader
from data.cleaner import DataCleaner
from analysis.features import FeatureExtractor
from analysis.timing import TimingAnalyzer
from analysis.statistics import StatisticalAnalyzer
from ml.clustering import ClusterAnalyzer
from visualization.plots import PlotGenerator
from visualization.signals import SignalVisualizer
from visualization.ml_plots import MLVisualizer
from utils.helpers import safe_plot_close, format_metric
from utils.report import ReportGenerator
from ui.sidebar import create_sidebar
from ui.tabs import (
    create_overview_tab, create_outlier_tab, create_timing_tab,
    create_features_tab, create_ml_tab, create_signals_tab,
    create_report_tab, create_anomaly_tab
)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {
            'X_scaled': None,
            'kmeans_labels': None,
            'dbscan_labels': None,
            'sil_score': 0,
            'pca': None,
            'X_pca': None,
            'features_df': None
        }
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'outlier_mask' not in st.session_state:
        st.session_state.outlier_mask = None

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Title
    st.title(PAGE_TITLE)
    st.markdown("Upload your CSV file to begin comprehensive analysis of gym movement sensor data.")
    
    # Initialize components
    data_loader = DataLoader()
    data_cleaner = DataCleaner()
    feature_extractor = FeatureExtractor()
    timing_analyzer = TimingAnalyzer()
    stats_analyzer = StatisticalAnalyzer()
    cluster_analyzer = ClusterAnalyzer()
    plot_generator = PlotGenerator()
    signal_visualizer = SignalVisualizer()
    ml_visualizer = MLVisualizer()
    report_generator = ReportGenerator()
    
    # Create sidebar
    uploaded_file = create_sidebar()
    
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Expected Data Format:
        - **Sensor columns**: `ax`, `ay`, `az`, `gx`, `gy`, `gz`
        - **Metadata**: `athlete_id`, `exercise_type`, `weight_kg`, `set_number`, `rep_number`
        - **Timestamp**: `timestamp` (optional but recommended)
        - **Sampling rate**: 100 Hz
        """)
        return
    
    # Load data
    with st.spinner("Loading data..."):
        df, error = data_loader.load_data(uploaded_file)
    
    if error:
        st.error(f"❌ Error loading file: {error}")
        st.stop()
        return
    
    # Add magnitudes
    with st.spinner("Computing sensor magnitudes..."):
        df = data_cleaner.compute_magnitudes(df)
    
    # Store in session state
    st.session_state.df = df
    st.session_state.data_loaded = True
    
    # Create tabs
    tabs = st.tabs(TABS)
    
    # Tab 1: Data Overview
    with tabs[0]:
        create_overview_tab(df, data_loader, stats_analyzer)
    
    # Tab 2: Outlier Detection
    with tabs[1]:
        outlier_mask = create_outlier_tab(df, data_cleaner, plot_generator)
        st.session_state.outlier_mask = outlier_mask
    
    # Tab 3: Timing Analysis
    with tabs[2]:
        create_timing_tab(df, timing_analyzer, plot_generator)
    
    # Tab 4: Feature Extraction
    with tabs[3]:
        features_df = create_features_tab(df, feature_extractor, stats_analyzer, plot_generator)
        if features_df is not None and not features_df.empty:
            st.session_state.ml_results['features_df'] = features_df
    
    # Tab 5: ML Analysis
    with tabs[4]:
        ml_results = create_ml_tab(
            st.session_state.ml_results['features_df'],
            cluster_analyzer,
            ml_visualizer
        )
        if ml_results:
            st.session_state.ml_results.update(ml_results)
    
    # Tab 6: Signal Plots
    with tabs[5]:
        create_signals_tab(df, signal_visualizer)
    
    # Tab 7: Summary Report
    with tabs[6]:
        create_report_tab(
            df,
            st.session_state.outlier_mask,
            st.session_state.ml_results,
            data_loader,
            stats_analyzer,
            report_generator
        )
    
    # Tab 8: Anomaly Investigation
    with tabs[7]:
        create_anomaly_tab(
            df,
            st.session_state.ml_results,
            timing_analyzer,
            cluster_analyzer,
            plot_generator
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error in application: {str(e)}")
        st.error("Please check your data format and try again.")
        if st.checkbox("Show detailed error"):
            st.exception(e)