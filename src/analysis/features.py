"""
Feature extraction module for sensor data analysis
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
import streamlit as st
from typing import Dict, List, Optional
from config import (
    WAVELET_TYPE, WAVELET_LEVEL, FFT_MIN_SAMPLES,
    SAMPLING_RATE_HZ, REQUIRED_SENSOR_COLS, METADATA_COLS
)


class FeatureExtractor:
    """Class for extracting features from sensor data"""
    
    def __init__(self, sampling_rate: int = SAMPLING_RATE_HZ):
        self.sampling_rate = sampling_rate
        self.wavelet_type = WAVELET_TYPE
        self.wavelet_level = WAVELET_LEVEL
    
    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract time-domain features from sensor data"""
        if len(data) == 0:
            return self._get_default_time_features()
        
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            return self._get_default_time_features()
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(data)
        features['std'] = np.std(data) if len(data) > 1 else 0
        features['min'] = np.min(data)
        features['max'] = np.max(data)
        features['range'] = features['max'] - features['min']
        
        # Higher order statistics
        features['skew'] = stats.skew(data) if len(data) > 2 else 0
        features['kurtosis'] = stats.kurtosis(data) if len(data) > 3 else 0
        
        # RMS and variance
        features['rms'] = np.sqrt(np.mean(data**2))
        features['variance'] = np.var(data)
        
        # Percentiles
        features['p25'] = np.percentile(data, 25)
        features['p50'] = np.percentile(data, 50)
        features['p75'] = np.percentile(data, 75)
        features['iqr'] = features['p75'] - features['p25']
        
        # Peak detection
        if len(data) > 3:
            try:
                threshold = np.mean(data) + np.std(data)
                peaks, _ = signal.find_peaks(data, height=threshold)
                features['peak_count'] = len(peaks)
                features['peak_mean_height'] = np.mean(data[peaks]) if len(peaks) > 0 else 0
            except:
                features['peak_count'] = 0
                features['peak_mean_height'] = 0
        else:
            features['peak_count'] = 0
            features['peak_mean_height'] = 0
        
        # Zero crossing rate
        features['zero_crossing_rate'] = self._calculate_zero_crossing_rate(data)
        
        # Mean absolute deviation
        features['mad'] = np.mean(np.abs(data - np.mean(data)))
        
        return features
    
    def extract_frequency_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features using FFT"""
        if len(data) < FFT_MIN_SAMPLES:
            return self._get_default_frequency_features()
        
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) < FFT_MIN_SAMPLES:
            return self._get_default_frequency_features()
        
        features = {}
        
        # Compute FFT
        n = len(data)
        fft_vals = fft(data)
        freqs = fftfreq(n, 1/self.sampling_rate)
        
        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        if len(fft_vals) > 0:
            # Dominant frequency
            dom_freq_idx = np.argmax(fft_vals)
            features['dominant_freq'] = freqs[dom_freq_idx]
            features['dominant_freq_magnitude'] = fft_vals[dom_freq_idx]
            
            # Power spectral density
            psd = (fft_vals**2) / n
            features['mean_psd'] = np.mean(psd)
            features['total_power'] = np.sum(psd)
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Frequency bands (0-10Hz, 10-20Hz, 20-30Hz, 30-40Hz, 40-50Hz)
            bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
            for i, (low, high) in enumerate(bands):
                band_mask = (freqs >= low) & (freqs < high)
                if np.any(band_mask):
                    features[f'band_power_{low}_{high}hz'] = np.sum(psd[band_mask])
                else:
                    features[f'band_power_{low}_{high}hz'] = 0
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
            
            # Spectral spread
            features['spectral_spread'] = np.sqrt(
                np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd)
            )
        else:
            features = self._get_default_frequency_features()
        
        return features
    
    def extract_wavelet_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract wavelet energy features"""
        min_samples = 2**self.wavelet_level
        
        if len(data) < min_samples:
            return self._get_default_wavelet_features()
        
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) < min_samples:
            return self._get_default_wavelet_features()
        
        features = {}
        
        try:
            # Adjust level if necessary
            max_level = pywt.dwt_max_level(len(data), self.wavelet_type)
            level = min(self.wavelet_level, max_level)
            
            if level < 1:
                features['wavelet_energy_L0'] = np.sum(data**2)
                return features
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(data, self.wavelet_type, level=level)
            
            # Calculate energy for each level
            total_energy = 0
            for i, c in enumerate(coeffs):
                energy = np.sum(c**2)
                features[f'wavelet_energy_L{i}'] = energy
                total_energy += energy
            
            # Calculate relative energy
            for i in range(len(coeffs)):
                features[f'wavelet_rel_energy_L{i}'] = features[f'wavelet_energy_L{i}'] / total_energy if total_energy > 0 else 0
            
            # Wavelet entropy
            rel_energies = [features[f'wavelet_rel_energy_L{i}'] for i in range(len(coeffs))]
            features['wavelet_entropy'] = -np.sum([e * np.log2(e + 1e-10) for e in rel_energies if e > 0])
            
        except Exception as e:
            features = self._get_default_wavelet_features()
        
        return features
    
    @st.cache_data
    def extract_all_features(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features for each repetition"""
        feature_list = []
        sensor_cols = REQUIRED_SENSOR_COLS + ['acc_mag', 'gyro_mag']
        
        # Check required grouping columns
        required_cols = METADATA_COLS
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        groups = df.groupby(required_cols)
        
        for name, group in groups:
            if len(group) == 0:
                continue
            
            features = {
                'athlete_id': name[0],
                'exercise_type': name[1],
                'weight_kg': name[2],
                'set_number': name[3],
                'rep_number': name[4]
            }
            
            # Extract features for each sensor
            for col in sensor_cols:
                if col in group.columns:
                    data = group[col].values
                    
                    # Skip if all values are NaN
                    if np.all(np.isnan(data)):
                        continue
                    
                    # Time-domain features
                    td_features = _self.extract_time_domain_features(data)
                    for feat_name, feat_val in td_features.items():
                        features[f'{col}_{feat_name}'] = feat_val
                    
                    # Frequency-domain features
                    fd_features = _self.extract_frequency_domain_features(data)
                    for feat_name, feat_val in fd_features.items():
                        features[f'{col}_{feat_name}'] = feat_val
                    
                    # Wavelet features
                    wv_features = _self.extract_wavelet_features(data)
                    for feat_name, feat_val in wv_features.items():
                        features[f'{col}_{feat_name}'] = feat_val
            
            # Add cross-sensor features
            if 'acc_mag' in group.columns and 'gyro_mag' in group.columns:
                acc_data = group['acc_mag'].values
                gyro_data = group['gyro_mag'].values
                
                # Correlation between acceleration and gyroscope
                if len(acc_data) > 1 and len(gyro_data) > 1:
                    features['acc_gyro_correlation'] = np.corrcoef(acc_data, gyro_data)[0, 1]
                else:
                    features['acc_gyro_correlation'] = 0
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def _calculate_zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        if len(data) < 2:
            return 0
        
        # Center the data
        data_centered = data - np.mean(data)
        
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(data_centered)) != 0)
        
        return zero_crossings / len(data)
    
    def _get_default_time_features(self) -> Dict[str, float]:
        """Return default time domain features when data is insufficient"""
        return {
            'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0,
            'skew': 0, 'kurtosis': 0, 'rms': 0, 'variance': 0,
            'p25': 0, 'p50': 0, 'p75': 0, 'iqr': 0,
            'peak_count': 0, 'peak_mean_height': 0,
            'zero_crossing_rate': 0, 'mad': 0
        }
    
    def _get_default_frequency_features(self) -> Dict[str, float]:
        """Return default frequency domain features when data is insufficient"""
        features = {
            'dominant_freq': 0, 'dominant_freq_magnitude': 0,
            'mean_psd': 0, 'total_power': 0, 'spectral_entropy': 0,
            'spectral_centroid': 0, 'spectral_spread': 0
        }
        
        # Add band powers
        bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
        for low, high in bands:
            features[f'band_power_{low}_{high}hz'] = 0
        
        return features
    
    def _get_default_wavelet_features(self) -> Dict[str, float]:
        """Return default wavelet features when data is insufficient"""
        features = {}
        for i in range(self.wavelet_level + 1):
            features[f'wavelet_energy_L{i}'] = 0
            features[f'wavelet_rel_energy_L{i}'] = 0
        features['wavelet_entropy'] = 0
        return features
    
    def get_feature_importance(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance based on variance"""
        # Exclude metadata columns
        feature_cols = [col for col in features_df.columns if col not in METADATA_COLS]
        
        if not feature_cols:
            return pd.DataFrame()
        
        # Calculate variance for each feature
        variances = features_df[feature_cols].var()
        
        # Calculate coefficient of variation
        means = features_df[feature_cols].mean()
        cv = variances / (means + 1e-10)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'variance': variances.values,
            'mean': means.values,
            'cv': cv.values
        })
        
        # Sort by coefficient of variation
        importance_df = importance_df.sort_values('cv', ascending=False)
        
        return importance_df