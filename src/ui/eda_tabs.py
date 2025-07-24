"""
Enhanced EDA tabs for Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict
from utils.helpers import format_metric, safe_plot_close


def create_signal_quality_tab(df: pd.DataFrame, signal_quality_analyzer, 
                            eda_visualizer, stats_tester):
    """Create comprehensive signal quality analysis tab"""
    st.header("🎯 Signal Quality & Drift Analysis")
    
    # Signal quality metrics
    st.subheader("Signal-to-Noise Ratio Analysis")
    
    snr_results = {}
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Accelerometer SNR (dB)**")
        for sensor in ['ax', 'ay', 'az']:
            if sensor in df.columns:
                snr = signal_quality_analyzer.calculate_snr(df[sensor].values)
                snr_results[sensor] = snr
                color = "🟢" if snr > 20 else "🟡" if snr > 10 else "🔴"
                st.write(f"{color} {sensor.upper()}: {snr:.1f} dB")
    
    with col2:
        st.write("**Gyroscope SNR (dB)**")
        for sensor in ['gx', 'gy', 'gz']:
            if sensor in df.columns:
                snr = signal_quality_analyzer.calculate_snr(df[sensor].values)
                snr_results[sensor] = snr
                color = "🟢" if snr > 20 else "🟡" if snr > 10 else "🔴"
                st.write(f"{color} {sensor.upper()}: {snr:.1f} dB")
    
    # Sensor drift analysis
    st.subheader("Sensor Drift Analysis")
    
    with st.spinner("Analyzing sensor drift..."):
        drift_results = signal_quality_analyzer.detect_sensor_drift(df)
    
    if drift_results:
        # Display drift metrics
        drift_df = pd.DataFrame(drift_results).T
        drift_df['drift_rate_abs'] = drift_df['drift_rate'].abs()
        
        # Highlight problematic sensors
        problematic_sensors = drift_df[drift_df['drift_rate_abs'] > 0.01].index.tolist()
        
        if problematic_sensors:
            st.warning(f"⚠️ Significant drift detected in sensors: {', '.join(problematic_sensors)}")
        else:
            st.success("✅ All sensors show minimal drift")
        
        # Drift visualization
        fig = eda_visualizer.plot_sensor_drift_analysis(drift_results)
        st.pyplot(fig)
        safe_plot_close(fig)
        
        # Detailed drift table
        with st.expander("View Detailed Drift Metrics"):
            st.dataframe(drift_df.round(4))
    
    # Noise characteristics
    st.subheader("Noise Characteristics")
    
    noise_analysis = {}
    for sensor in sensor_cols:
        if sensor in df.columns:
            noise_analysis[sensor] = signal_quality_analyzer.analyze_noise_characteristics(
                df[sensor].values
            )
    
    if noise_analysis:
        noise_df = pd.DataFrame(noise_analysis).T
        
        # Create noise summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_noise_rms = noise_df['noise_rms'].mean()
            st.metric("Avg Noise RMS", f"{avg_noise_rms:.4f}")
        with col2:
            max_crest = noise_df['crest_factor'].max()
            st.metric("Max Crest Factor", f"{max_crest:.2f}")
        with col3:
            avg_kurtosis = noise_df['noise_distribution_kurtosis'].mean()
            st.metric("Avg Noise Kurtosis", f"{avg_kurtosis:.2f}")
        with col4:
            quality_score = 100 - min(100, avg_noise_rms * 1000)
            st.metric("Quality Score", f"{quality_score:.0f}/100")
        
        # Noise distribution analysis
        st.write("**Noise Distribution Analysis**")
        st.dataframe(noise_df.round(3))
    
    # Signal stability across repetitions
    if all(col in df.columns for col in ['athlete_id', 'exercise_type', 'rep_number']):
        st.subheader("Signal Stability Analysis")
        
        with st.spinner("Analyzing signal stability..."):
            stability_df = signal_quality_analyzer.calculate_signal_stability(
                df, ['athlete_id', 'exercise_type', 'rep_number']
            )
        
        if not stability_df.empty:
            # Summary statistics
            stable_count = sum(stability_df[[col for col in stability_df.columns 
                                            if '_is_stationary' in col]].sum())
            total_tests = len([col for col in stability_df.columns 
                              if '_is_stationary' in col]) * len(stability_df)
            
            stability_pct = (stable_count / total_tests * 100) if total_tests > 0 else 0
            
            st.info(f"📊 Signal Stability: {stability_pct:.1f}% of signals are stationary")
            
            # Exercise-wise stability
            if 'exercise_type' in stability_df.columns:
                exercise_stability = stability_df.groupby('exercise_type').agg({
                    col: 'mean' for col in stability_df.columns if '_cv' in col
                })
                
                st.write("**Coefficient of Variation by Exercise**")
                st.dataframe(exercise_stability.round(3))


def create_advanced_visualization_tab(df: pd.DataFrame, eda_visualizer, stats_tester):
    """Create advanced visualization tab with interactive plots"""
    st.header("📊 Advanced Signal Visualizations")
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["3D Trajectory", "Comprehensive Spectrogram", "Autocorrelation Analysis",
         "Cross-Sensor Correlation", "Movement Patterns"]
    )
    
    if viz_type == "3D Trajectory":
        st.subheader("Interactive 3D Sensor Trajectories")
        
        # Sampling controls
        col1, col2 = st.columns(2)
        with col1:
            sample_fraction = st.slider("Data sampling fraction", 0.01, 1.0, 0.1,
                                      help="Reduce for better performance")
        with col2:
            athlete_filter = st.selectbox("Filter by athlete", 
                                        ["All"] + list(df['athlete_id'].unique()))
        
        # Filter data
        plot_df = df.copy()
        if athlete_filter != "All":
            plot_df = plot_df[plot_df['athlete_id'] == athlete_filter]
        
        # Create 3D plot
        fig = eda_visualizer.create_interactive_3d_trajectory(plot_df, sample_fraction)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Tip: Use mouse to rotate and zoom the 3D plot")
    
    elif viz_type == "Comprehensive Spectrogram":
        st.subheader("Multi-View Frequency Analysis")
        
        # Sensor selector
        sensor = st.selectbox("Select Sensor", 
                            [col for col in df.columns if col in 
                             ['ax', 'ay', 'az', 'gx', 'gy', 'gz']])
        
        # Optional filtering
        if st.checkbox("Filter by specific repetition"):
            col1, col2, col3 = st.columns(3)
            with col1:
                athlete = st.selectbox("Athlete", df['athlete_id'].unique())
            with col2:
                exercise = st.selectbox("Exercise", 
                                      df[df['athlete_id'] == athlete]['exercise_type'].unique())
            with col3:
                rep = st.selectbox("Rep", 
                                 df[(df['athlete_id'] == athlete) & 
                                    (df['exercise_type'] == exercise)]['rep_number'].unique())
            
            plot_df = df[(df['athlete_id'] == athlete) & 
                        (df['exercise_type'] == exercise) & 
                        (df['rep_number'] == rep)]
        else:
            plot_df = df
        
        # Create spectrogram
        fig = eda_visualizer.plot_comprehensive_spectrogram(plot_df, sensor)
        st.pyplot(fig)
        safe_plot_close(fig)
    
    elif viz_type == "Autocorrelation Analysis":
        st.subheader("Signal Autocorrelation Analysis")
        
        max_lag = st.slider("Maximum lag (samples)", 50, 500, 200)
        
        # Create autocorrelation plot
        fig = eda_visualizer.plot_autocorrelation_analysis(df, max_lag)
        st.pyplot(fig)
        safe_plot_close(fig)
        
        st.info("🔍 Autocorrelation reveals periodic patterns and signal memory")
    
    elif viz_type == "Cross-Sensor Correlation":
        st.subheader("Interactive Sensor Correlation Matrix")
        
        # Create correlation heatmap
        fig = eda_visualizer.create_sensor_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical independence tests
        if st.checkbox("Show statistical independence tests"):
            with st.spinner("Running independence tests..."):
                independence_results = stats_tester.test_sensor_independence(df)
            
            if independence_results:
                indep_df = pd.DataFrame(independence_results).T
                st.dataframe(indep_df.round(3))
    
    elif viz_type == "Movement Patterns":
        st.subheader("Movement Pattern Analysis")
        
        window_size = st.slider("Analysis window size", 10, 200, 50)
        
        # Create movement pattern plot
        fig = eda_visualizer.plot_movement_patterns(df, window_size)
        st.pyplot(fig)
        safe_plot_close(fig)


def create_statistical_testing_tab(df: pd.DataFrame, stats_tester):
    """Create statistical testing tab"""
    st.header("📈 Statistical Testing & Analysis")
    
    # Test selector
    test_type = st.selectbox(
        "Select Statistical Test",
        ["Stationarity Tests", "Normality Tests", "Exercise Comparison", 
         "Sensor Independence"]
    )
    
    if test_type == "Stationarity Tests":
        st.subheader("Signal Stationarity Analysis")
        
        sensor = st.selectbox("Select sensor for analysis",
                            [col for col in df.columns if col in 
                             ['ax', 'ay', 'az', 'gx', 'gy', 'gz']])
        
        if st.button("Run Stationarity Tests"):
            with st.spinner("Running tests..."):
                results = stats_tester.test_stationarity(df[sensor])
            
            if 'error' not in results:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'adf' in results:
                        st.write("**Augmented Dickey-Fuller Test**")
                        st.write(f"Statistic: {results['adf']['statistic']:.4f}")
                        st.write(f"p-value: {results['adf']['p_value']:.4f}")
                        st.write(f"Result: {results['adf']['interpretation']}")
                
                with col2:
                    if 'kpss' in results:
                        st.write("**KPSS Test**")
                        st.write(f"Statistic: {results['kpss']['statistic']:.4f}")
                        st.write(f"p-value: {results['kpss']['p_value']:.4f}")
                        st.write(f"Result: {results['kpss']['interpretation']}")
                
                # Interpretation
                if 'adf' in results and 'kpss' in results:
                    if results['adf']['is_stationary'] and results['kpss']['is_stationary']:
                        st.success("✅ Both tests indicate the signal is stationary")
                    elif not results['adf']['is_stationary'] and not results['kpss']['is_stationary']:
                        st.error("❌ Both tests indicate the signal is non-stationary")
                    else:
                        st.warning("⚠️ Tests show conflicting results - further investigation needed")
    
    elif test_type == "Normality Tests":
        st.subheader("Distribution Normality Testing")
        
        sensor = st.selectbox("Select sensor for normality test",
                            [col for col in df.columns if col in 
                             ['ax', 'ay', 'az', 'gx', 'gy', 'gz']])
        
        if st.button("Run Normality Tests"):
            with st.spinner("Testing normality..."):
                results = stats_tester.test_normality(df[sensor])
            
            if 'error' not in results:
                # Display results
                normal_count = sum(1 for test in results.values() 
                                 if isinstance(test, dict) and test.get('is_normal', False))
                total_tests = sum(1 for test in results.values() 
                                if isinstance(test, dict) and 'is_normal' in test)
                
                st.info(f"📊 {normal_count}/{total_tests} tests indicate normal distribution")
                
                # Detailed results
                with st.expander("View Detailed Test Results"):
                    for test_name, test_results in results.items():
                        if isinstance(test_results, dict):
                            st.write(f"**{test_name.replace('_', ' ').title()}**")
                            st.json(test_results)
                
                # Q-Q correlation
                if 'qq_correlation' in results:
                    qq_corr = results['qq_correlation']
                    interpretation = "Strong" if qq_corr > 0.98 else "Moderate" if qq_corr > 0.95 else "Weak"
                    st.write(f"**Q-Q Plot Correlation**: {qq_corr:.4f} ({interpretation} normality)")
    
    elif test_type == "Exercise Comparison":
        st.subheader("Statistical Comparison Across Exercises")
        
        sensor = st.selectbox("Select sensor for comparison",
                            [col for col in df.columns if col in 
                             ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']])
        
        if st.button("Compare Exercises"):
            with st.spinner("Running statistical comparisons..."):
                results = stats_tester.compare_exercise_patterns(df, sensor)
            
            if 'error' not in results:
                # ANOVA results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**One-way ANOVA**")
                    st.write(f"F-statistic: {results['anova']['f_statistic']:.4f}")
                    st.write(f"p-value: {results['anova']['p_value']:.4f}")
                    if results['anova']['significant']:
                        st.success("✅ Significant differences found")
                    else:
                        st.info("No significant differences")
                
                with col2:
                    st.write("**Kruskal-Wallis Test**")
                    st.write(f"H-statistic: {results['kruskal_wallis']['h_statistic']:.4f}")
                    st.write(f"p-value: {results['kruskal_wallis']['p_value']:.4f}")
                    if results['kruskal_wallis']['significant']:
                        st.success("✅ Significant differences found")
                    else:
                        st.info("No significant differences")
                
                # Post-hoc analysis
                if 'tukey_hsd' in results and results['tukey_hsd']['significant_pairs']:
                    st.write("**Significant Pairwise Differences (Tukey HSD)**")
                    pairs_df = pd.DataFrame(results['tukey_hsd']['significant_pairs'])
                    st.dataframe(pairs_df)