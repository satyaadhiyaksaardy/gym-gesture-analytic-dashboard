# src/analysis/signal_quality.py - Update imports and fix the method
"""
Signal quality analysis module for IMU sensor data
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from typing import Dict, Tuple, Optional
from config import SAMPLING_RATE_HZ


class SignalQualityAnalyzer:
    """Class for analyzing signal quality metrics"""
    
    def __init__(self, sampling_rate: int = SAMPLING_RATE_HZ):
        self.sampling_rate = sampling_rate
    
    @st.cache_data
    def calculate_snr(_self, data: np.ndarray, noise_floor_percentile: float = 5) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            data: Signal data
            noise_floor_percentile: Percentile to estimate noise floor
            
        Returns:
            SNR in dB
        """
        if len(data) < 10:
            return 0.0
        
        # Remove DC component
        data_ac = data - np.mean(data)
        
        # Estimate noise floor using lower percentile of frequency spectrum
        fft_vals = np.abs(fft(data_ac))
        noise_floor = np.percentile(fft_vals, noise_floor_percentile)
        
        # Calculate signal power
        signal_power = np.mean(data_ac**2)
        noise_power = noise_floor**2
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return float(snr_db)
        return 0.0
    
    @st.cache_data
    def detect_sensor_drift(_self, df: pd.DataFrame, window_size: int = 100) -> Dict:
        """
        Detect sensor drift over time
        
        Args:
            df: DataFrame with sensor data
            window_size: Window size for drift calculation
            
        Returns:
            Dictionary with drift metrics
        """
        drift_results = {}
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        for col in sensor_cols:
            if col in df.columns:
                data = df[col].values
                
                # Calculate rolling baseline
                if len(data) > window_size:
                    rolling_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
                    
                    # Drift is the trend in the rolling mean
                    time_indices = np.arange(len(rolling_mean))
                    valid_idx = ~np.isnan(rolling_mean)
                    
                    if np.sum(valid_idx) > 10:
                        # Linear regression to detect drift
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            time_indices[valid_idx], rolling_mean[valid_idx]
                        )
                        
                        drift_results[col] = {
                            'drift_rate': slope * _self.sampling_rate,  # drift per second
                            'total_drift': slope * len(data),
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'baseline_std': np.std(rolling_mean[valid_idx])
                        }
                    else:
                        drift_results[col] = _self._default_drift_metrics()
                else:
                    drift_results[col] = _self._default_drift_metrics()
        
        return drift_results
    
    def _default_drift_metrics(self) -> Dict:
        """Return default drift metrics"""
        return {
            'drift_rate': 0.0,
            'total_drift': 0.0,
            'r_squared': 0.0,
            'p_value': 1.0,
            'baseline_std': 0.0
        }
    
    @st.cache_data
    def analyze_noise_characteristics(_self, data: np.ndarray) -> Dict:
        """
        Analyze noise characteristics of the signal
        
        Returns:
            Dictionary with noise metrics
        """
        if len(data) < 100:
            return {}
        
        # High-pass filter to isolate noise
        sos = signal.butter(4, 20, 'hp', fs=_self.sampling_rate, output='sos')
        noise = signal.sosfilt(sos, data)
        
        # Noise statistics
        return {
            'noise_rms': np.sqrt(np.mean(noise**2)),
            'noise_peak_to_peak': np.max(noise) - np.min(noise),
            'noise_variance': np.var(noise),
            'crest_factor': np.max(np.abs(noise)) / np.sqrt(np.mean(noise**2)) if np.mean(noise**2) > 0 else 0,
            'noise_distribution_kurtosis': stats.kurtosis(noise),
            'noise_distribution_skewness': stats.skew(noise)
        }
    
    @st.cache_data
    def calculate_signal_stability(_self, df: pd.DataFrame, rep_grouping_cols: list) -> pd.DataFrame:
        """
        Calculate signal stability metrics across repetitions
        
        Returns:
            DataFrame with stability metrics per group
        """
        stability_results = []
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        groups = df.groupby(rep_grouping_cols)
        
        for name, group in groups:
            result = {col: name[i] for i, col in enumerate(rep_grouping_cols)}
            
            for sensor in sensor_cols:
                if sensor in group.columns:
                    data = group[sensor].values
                    
                    if len(data) > 10:
                        # Stationarity test using adfuller from statsmodels
                        try:
                            adf_result = adfuller(data, autolag='AIC')
                            adf_stat = adf_result[0]
                            adf_pvalue = adf_result[1]
                        except Exception as e:
                            # Fallback if ADF test fails
                            adf_stat = 0
                            adf_pvalue = 1.0
                        
                        # Coefficient of variation
                        cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
                        
                        result.update({
                            f'{sensor}_cv': cv,
                            f'{sensor}_adf_statistic': adf_stat,
                            f'{sensor}_is_stationary': adf_pvalue < 0.05,
                            f'{sensor}_range_ratio': (np.max(data) - np.min(data)) / np.std(data) if np.std(data) > 0 else 0
                        })
            
            stability_results.append(result)
        
        return pd.DataFrame(stability_results)