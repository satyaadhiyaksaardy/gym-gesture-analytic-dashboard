"""
Tab components for the UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict
from utils.helpers import format_metric, safe_plot_close


def create_overview_tab(df: pd.DataFrame, data_loader, stats_analyzer):
    """Create the data overview tab"""
    st.header("Data Overview & Validation")
    
    # Get data info
    data_info = data_loader.get_data_info(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", format_metric(data_info['total_rows'], 'count'))
    with col2:
        st.metric("Total Columns", data_info['total_columns'])
    with col3:
        st.metric("Memory Usage", format_metric(data_info['memory_usage_mb'], 'memory_mb'))
    
    # Data types and missing values
    st.subheader("Data Types & Missing Values")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    else:
        st.warning("No numeric columns found for statistical analysis")
    
    # Additional info
    if data_info.get('unique_athlete_id'):
        st.info(f"Number of athletes: {data_info['unique_athlete_id']}")
    if data_info.get('unique_exercise_type'):
        st.info(f"Number of exercise types: {data_info['unique_exercise_type']}")


def create_outlier_tab(df: pd.DataFrame, data_cleaner, plot_generator) -> pd.Series:
    """Create the outlier detection tab"""
    st.header("Outlier Detection & Cleaning")
    
    # Detect outliers
    with st.spinner("Detecting outliers..."):
        outlier_mask = data_cleaner.detect_outliers(df)
    
    # Get statistics
    outlier_stats = data_cleaner.get_outlier_statistics(df, outlier_mask)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outliers Detected", format_metric(outlier_stats['total_outliers'], 'count'))
    with col2:
        st.metric("Outlier Percentage", format_metric(outlier_stats['outlier_percentage'], 'percentage'))
    
    # Visualization
    if outlier_stats['outliers_by_sensor']:
        st.subheader("Outliers by Sensor")
        fig = plot_generator.plot_outliers_by_sensor(outlier_stats['outliers_by_sensor'])
        st.pyplot(fig)
        safe_plot_close(fig)

    cleaned_df = data_cleaner.clean_data(df, outlier_mask)
    if 'duration_ms' in df.columns and 'duration_ms' in cleaned_df.columns:
        fig = plot_generator.plot_distribution_comparison(df['duration_ms'], cleaned_df['duration_ms'],
                                                         label1="Raw", label2="Cleaned",
                                                         title="Duration Distribution: Raw vs. Cleaned")
        st.pyplot(fig)
        safe_plot_close(fig)

        # Histogram for raw and cleaned durations
        fig_raw = plot_generator.plot_duration_histogram(df['duration_ms'])
        st.pyplot(fig_raw)
        safe_plot_close(fig_raw)
        fig_clean = plot_generator.plot_duration_histogram(cleaned_df['duration_ms'])
        st.pyplot(fig_clean)
        safe_plot_close(fig_clean)
    
    # Data view options
    view_option = st.radio("Data View", ["Raw Data", "Cleaned Data (Outliers Removed)"])
    
    if view_option == "Raw Data":
        display_df = df
        st.info(f"Showing raw data with {len(df):,} rows")
    else:
        display_df = data_cleaner.clean_data(df, outlier_mask)
        st.success(f"Showing cleaned data with {len(display_df):,} rows")
    
    # Show sample
    st.subheader("Data Sample")
    sample_size = min(100, len(display_df))
    st.dataframe(display_df.head(sample_size), use_container_width=True)
    
    # Download option
    if st.button("💾 Download Cleaned Data"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    
    return outlier_mask


def create_timing_tab(df: pd.DataFrame, timing_analyzer, plot_generator):
    """Create the timing analysis tab"""
    st.header("Timing & Sampling Analysis")
    
    # Check if timestamp exists
    if 'timestamp' not in df.columns:
        st.warning("Timestamp column not found. Timing analysis requires timestamp data.")
        return
    
    # Analyze timing patterns
    timing_patterns = timing_analyzer.analyze_timing_patterns(df)
    
    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Timing Quality", timing_patterns['timing_quality'])
    with col2:
        st.metric("Overall Sampling Rate", 
                 f"{timing_patterns.get('overall_sampling_rate', 0):.1f} Hz")
    with col3:
        st.metric("Irregular Sampling", 
                 format_metric(timing_patterns.get('irregular_sampling_rate', 0), 'percentage'))
    with col4:
        st.metric("Total Duration", 
                 f"{timing_patterns.get('total_duration_ms', 0)/1000:.1f} s")
    
    # Sampling consistency
    st.subheader("Sampling Consistency Analysis")
    with st.spinner("Analyzing sampling consistency..."):
        sampling_df = timing_analyzer.analyze_sampling_consistency(df)
    
    if not sampling_df.empty:
        # Calculate metrics
        sampling_metrics = timing_analyzer.calculate_sampling_metrics(sampling_df)
        
        # Display summary
        st.write("**Sampling Quality Score:**", 
                f"{sampling_metrics.get('sampling_quality_score', 0):.1f}/100")
        
        # Show problematic groups
        irregular_groups = timing_analyzer.identify_irregular_sampling(sampling_df)
        if not irregular_groups.empty:
            st.warning(f"Found {len(irregular_groups)} groups with irregular sampling")
            st.dataframe(irregular_groups.head(10), use_container_width=True)

    # Sampling intervals plot
    if not sampling_df.empty and 'mean_interval_ms' in sampling_df.columns:
        fig = plot_generator.plot_sampling_intervals(sampling_df['mean_interval_ms'])
        st.pyplot(fig)
        safe_plot_close(fig)

    # Correlation matrix for timing metrics
    numeric_cols = [c for c in sampling_df.columns if sampling_df[c].dtype in [float, int]]
    # Drop sample_count column if it exists
    if 'sample_count' in numeric_cols:
        numeric_cols.remove('sample_count')
    if len(numeric_cols) > 1:
        fig = plot_generator.plot_correlation_matrix(sampling_df[numeric_cols])
        st.pyplot(fig)
        safe_plot_close(fig)
    
    # Repetition durations
    st.subheader("Repetition Duration Analysis")
    with st.spinner("Computing repetition durations..."):
        rep_duration_df = timing_analyzer.compute_rep_durations(df)
    
    if not rep_duration_df.empty:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Duration", 
                     format_metric(rep_duration_df['duration_ms'].mean(), 'time_ms'))
        with col2:
            st.metric("Std Duration", 
                     format_metric(rep_duration_df['duration_ms'].std(), 'time_ms'))
        with col3:
            st.metric("Min Duration", 
                     format_metric(rep_duration_df['duration_ms'].min(), 'time_ms'))
        with col4:
            st.metric("Max Duration", 
                     format_metric(rep_duration_df['duration_ms'].max(), 'time_ms'))
        
        # Duration histogram
        if len(rep_duration_df) > 5:
            fig = plot_generator.plot_duration_histogram(rep_duration_df['duration_ms'])
            st.pyplot(fig)
            safe_plot_close(fig)


def create_features_tab(df: pd.DataFrame, feature_extractor, stats_analyzer, plot_generator) -> Optional[pd.DataFrame]:
    """Create the feature extraction tab"""
    st.header("Advanced Feature Extraction")
    
    # Check required columns
    required_cols = ['athlete_id', 'exercise_type', 'weight_kg', 'set_number', 'rep_number']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns for feature extraction: {', '.join(missing_cols)}")
        return None
    
    # Extract features
    with st.spinner("Extracting features... This may take a moment."):
        features_df = feature_extractor.extract_all_features(df)
    
    if features_df.empty:
        st.error("Failed to extract features. Please check your data format.")
        return None
    
    st.success(f"✅ Extracted {len(features_df.columns)-5} features from {len(features_df)} repetitions")
    
    # Display features
    st.subheader("Extracted Features Sample")
    st.dataframe(features_df.head(20), use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    importance_df = feature_extractor.get_feature_importance(features_df)
    if not importance_df.empty:
        fig = plot_generator.plot_feature_importance(importance_df)
        st.pyplot(fig)
        safe_plot_close(fig)
    
    # Correlation analysis
    st.subheader("Feature Correlation Analysis")
    feature_cols = [col for col in features_df.columns if col not in required_cols]
    if len(feature_cols) > 1:
        # Select subset for visualization
        selected_features = st.multiselect(
            "Select features for correlation analysis",
            feature_cols[:20],  # Show first 20 features
            default=feature_cols[:10]  # Default to first 10
        )
        
        if selected_features:
            fig = plot_generator.plot_correlation_matrix(features_df, selected_features)
            st.pyplot(fig)
            safe_plot_close(fig)
    
    # Download features
    if st.button("💾 Download Extracted Features"):
        csv = features_df.to_csv(index=False)
        st.download_button(
            label="Download Features CSV",
            data=csv,
            file_name="extracted_features.csv",
            mime="text/csv"
        )
    
    return features_df


def create_ml_tab(features_df: Optional[pd.DataFrame], cluster_analyzer, ml_visualizer) -> Optional[Dict]:
    """Create the machine learning analysis tab"""
    st.header("🤖 Machine Learning Analysis")
    
    if features_df is None or features_df.empty:
        st.warning("Please extract features first in the Feature Extraction tab")
        return None
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        eps_pct = st.slider("DBSCAN eps percentile", 50, 100, 80,
                           help="Percentile of k-distance for eps calculation")
    with col2:
        min_samp = st.slider("DBSCAN min_samples", 2, 50, 5,
                            help="Minimum samples for core points")
    with col3:
        k_max = st.slider("Max k for K-Means", 2, 10, 6,
                         help="Maximum number of clusters to test")
    
    # Run clustering
    if st.button("🔄 Run Clustering Analysis"):
        with st.spinner("Performing clustering analysis..."):
            results = cluster_analyzer.perform_clustering(
                features_df,
                eps_percentile=eps_pct,
                min_samples=min_samp,
                k_max=k_max
            )
        
        if results['X_scaled'] is None:
            st.error("Clustering failed. Please check your data has sufficient samples.")
            return None
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", format_metric(results['sil_score'], 'score'))
        with col2:
            st.metric("K-Means Clusters", results['best_k'])
        with col3:
            noise_count = (results['dbscan_labels'] == -1).sum() if results['dbscan_labels'] is not None else 0
            st.metric("DBSCAN Noise Points", noise_count)
        
        # Cluster statistics
        st.subheader("Cluster Statistics")
        
        # K-Means statistics
        if results['kmeans_labels'] is not None:
            kmeans_stats = cluster_analyzer.get_cluster_statistics(features_df, results['kmeans_labels'])
            if not kmeans_stats.empty:
                st.write("**K-Means Cluster Distribution:**")
                st.dataframe(kmeans_stats, use_container_width=True)
        
        # PCA Analysis
        st.subheader("PCA Analysis")
        with st.spinner("Performing PCA..."):
            pca, X_pca = cluster_analyzer.perform_pca(results['X_scaled'])
        
        if pca is not None:
            # Store PCA results
            results['pca'] = pca
            results['X_pca'] = X_pca
            
            # Display explained variance
            exp_var = pca.explained_variance_ratio_
            st.write("**Explained Variance Ratios:**")
            for i, var in enumerate(exp_var):
                st.write(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
            
            # Visualizations would go here
            st.info("PCA visualizations would be displayed here")

        # Silhouette analysis plot
        if results.get('silhouette_scores'):
            fig = ml_visualizer.plot_silhouette_analysis(results['silhouette_scores'])
            st.pyplot(fig)
            safe_plot_close(fig)

        # Cluster distribution
        if results.get('kmeans_labels') is not None and results.get('dbscan_labels') is not None:
            fig = ml_visualizer.plot_cluster_distribution(results['kmeans_labels'], results['dbscan_labels'])
            st.pyplot(fig)
            safe_plot_close(fig)

        # Cluster heatmap
        if results.get('kmeans_stats') is not None:
            fig = ml_visualizer.plot_cluster_heatmap(results['kmeans_stats'])
            st.pyplot(fig)
            safe_plot_close(fig)

        # PCA explained variance
        if results.get('pca') is not None and results.get('X_pca') is not None:
            exp_var = results['pca'].explained_variance_ratio_
            fig = ml_visualizer.plot_explained_variance(exp_var)
            st.pyplot(fig)
            safe_plot_close(fig)
            # Optionally add 2D/3D PCA plots
            fig2d = ml_visualizer.plot_pca_2d(results['X_pca'], results['kmeans_labels'], exp_var)
            st.pyplot(fig2d)
            safe_plot_close(fig2d)
            if results['X_pca'].shape[1] >= 3:
                fig3d = ml_visualizer.plot_pca_3d(results['X_pca'], results['kmeans_labels'], exp_var)
                st.pyplot(fig3d)
                safe_plot_close(fig3d)
        
        return results
    
    return None


def create_signals_tab(df: pd.DataFrame, signal_visualizer):
    """Create the signal visualization tab"""
    st.header("Signal Analysis & Visualizations")
    
    # Check required columns
    required_cols = ['athlete_id', 'exercise_type', 'rep_number']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Missing columns for signal visualization: {', '.join(missing_cols)}")
        return
    
    # Selection controls
    st.subheader("Select a Sample for Visualization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        athletes = sorted(df['athlete_id'].unique())
        selected_athlete = st.selectbox("Select Athlete", athletes)
    
    athlete_df = df[df['athlete_id'] == selected_athlete]
    
    with col2:
        exercises = sorted(athlete_df['exercise_type'].unique())
        selected_exercise = st.selectbox("Select Exercise", exercises)
    
    exercise_df = athlete_df[athlete_df['exercise_type'] == selected_exercise]
    
    with col3:
        reps = sorted(exercise_df['rep_number'].unique())
        selected_rep = st.selectbox("Select Rep Number", reps)
    
    # Get sample data
    sample_data = exercise_df[exercise_df['rep_number'] == selected_rep].copy()
    
    if 'timestamp' in sample_data.columns:
        sample_data = sample_data.sort_values('timestamp')
    
    if len(sample_data) == 0:
        st.warning("No data found for the selected combination")
        return
    
    st.info(f"Showing {len(sample_data)} samples for the selected repetition")
    
    # Signal plots
    st.subheader("Sensor Signals")
    
    # Accelerometer plot
    fig_acc = signal_visualizer.plot_sensor_signals(sample_data, 'accelerometer')
    st.pyplot(fig_acc)
    safe_plot_close(fig_acc)
    
    # Gyroscope plot
    fig_gyro = signal_visualizer.plot_sensor_signals(sample_data, 'gyroscope')
    st.pyplot(fig_gyro)
    safe_plot_close(fig_gyro)
    
    # Magnitude plots
    if 'acc_mag' in sample_data.columns or 'gyro_mag' in sample_data.columns:
        st.subheader("Magnitude Signals")
        fig_mag = signal_visualizer.plot_magnitude_signals(sample_data)
        st.pyplot(fig_mag)
        safe_plot_close(fig_mag)


def create_report_tab(df: pd.DataFrame, outlier_mask: Optional[pd.Series], 
                     ml_results: Dict, data_loader, stats_analyzer, report_generator, timing_analyzer):
    """Create the summary report tab"""
    st.header("📋 Summary Report")
    
    # Generate report content
    report_content = report_generator.generate_report(
        df, outlier_mask, ml_results, data_loader, stats_analyzer, timing_analyzer
    )
    
    # Display report preview
    st.text_area("Report Preview", report_content, height=400)
    
    # Download button
    st.download_button(
        label="📄 Download Full Report",
        data=report_content,
        file_name=report_generator.get_report_filename(),
        mime="text/plain"
    )


def create_anomaly_tab(df: pd.DataFrame, ml_results: Dict, 
                      timing_analyzer, cluster_analyzer, plot_generator, ml_visualizer):
    """Create the anomaly investigation tab"""
    st.header("🔍 Anomaly Investigation")
    st.markdown("Automated investigation of DBSCAN noise points and irregular sampling patterns")
    
    # DBSCAN Noise Analysis
    st.subheader("1. DBSCAN Noise Points Analysis")
    
    if ml_results.get('dbscan_labels') is not None and ml_results.get('features_df') is not None:
        noise_analysis = cluster_analyzer.analyze_dbscan_noise(
            ml_results['features_df'], 
            ml_results['dbscan_labels']
        )
        
        if noise_analysis['noise_count'] > 0:
            st.info(f"Found {noise_analysis['noise_count']} noise points "
                   f"({noise_analysis['noise_percentage']:.1f}% of total)")
            
            # Display noise distribution
            st.write("**Noise Distribution by Exercise Type:**")
            if noise_analysis['noise_by_exercise']:
                noise_df = pd.DataFrame.from_dict(
                    noise_analysis['noise_by_exercise'], 
                    orient='index'
                )
                st.dataframe(noise_df, use_container_width=True)

                # Exercise distribution plot for noise
                if 'sum' in noise_df.columns:
                    fig = plot_generator.plot_exercise_distribution(noise_df['sum'])
                    st.pyplot(fig)
                    safe_plot_close(fig)

            fig = ml_visualizer.plot_dbscan_noise_analysis(ml_results['features_df'], ml_results['dbscan_labels'])
            st.pyplot(fig)
            safe_plot_close(fig)
        else:
            st.success("No noise points detected in DBSCAN clustering")
    else:
        st.warning("Please run ML Analysis first to investigate DBSCAN noise points")
    
    # Irregular Sampling Analysis
    st.subheader("2. Irregular Sampling Pattern Analysis")
    
    if 'timestamp' in df.columns:
        with st.spinner("Analyzing irregular sampling patterns..."):
            sampling_df = timing_analyzer.analyze_sampling_consistency(df)
        
        if not sampling_df.empty:
            irregular_groups = timing_analyzer.identify_irregular_sampling(sampling_df)
            
            if not irregular_groups.empty:
                st.warning(f"Found {len(irregular_groups)} groups with irregular sampling")
                st.dataframe(irregular_groups.head(10), use_container_width=True)

                fig = plot_generator.plot_sampling_intervals(irregular_groups['mean_interval_ms'])
                st.pyplot(fig)
                safe_plot_close(fig)
            else:
                st.success("No irregular sampling patterns detected")
    else:
        st.info("Timestamp column not found. Sampling analysis requires timestamp data.")