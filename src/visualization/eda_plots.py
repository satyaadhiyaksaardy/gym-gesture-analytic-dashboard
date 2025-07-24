"""
Enhanced EDA visualization module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import welch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
from config import PLOT_STYLE, FIGURE_DPI, SAMPLING_RATE_HZ


class EDAVisualizer:
    """Enhanced visualizations for exploratory data analysis"""
    
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.dpi = FIGURE_DPI
        self.sampling_rate = SAMPLING_RATE_HZ
    
    def plot_sensor_drift_analysis(self, drift_results: dict) -> plt.Figure:
        """Visualize sensor drift analysis results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.dpi)
        
        sensors = list(drift_results.keys())
        drift_rates = [drift_results[s]['drift_rate'] for s in sensors]
        r_squared = [drift_results[s]['r_squared'] for s in sensors]
        
        # Drift rates
        colors = ['red' if abs(dr) > 0.01 else 'green' for dr in drift_rates]
        ax1.bar(sensors, drift_rates, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Drift Rate (units/second)')
        ax1.set_title('Sensor Drift Rates')
        ax1.grid(True, alpha=0.3)
        
        # R-squared values
        ax2.bar(sensors, r_squared, color='blue', alpha=0.7)
        ax2.set_ylabel('R² Value')
        ax2.set_xlabel('Sensor')
        ax2.set_title('Drift Linearity (R² of linear fit)')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_trajectory(self, df: pd.DataFrame, 
                                       sample_fraction: float = 0.1) -> go.Figure:
        """Create interactive 3D trajectory plot using Plotly"""
        # Downsample for performance
        step = max(1, int(1 / sample_fraction))
        df_sample = df.iloc[::step]
        
        fig = go.Figure()
        
        # Accelerometer trajectory
        if all(col in df_sample.columns for col in ['ax', 'ay', 'az']):
            fig.add_trace(go.Scatter3d(
                x=df_sample['ax'],
                y=df_sample['ay'],
                z=df_sample['az'],
                mode='lines+markers',
                name='Accelerometer',
                line=dict(color='blue', width=2),
                marker=dict(size=3, color=np.arange(len(df_sample)), colorscale='Viridis')
            ))
        
        # Gyroscope trajectory
        if all(col in df_sample.columns for col in ['gx', 'gy', 'gz']):
            fig.add_trace(go.Scatter3d(
                x=df_sample['gx'],
                y=df_sample['gy'],
                z=df_sample['gz'],
                mode='lines+markers',
                name='Gyroscope',
                line=dict(color='red', width=2),
                marker=dict(size=3, color=np.arange(len(df_sample)), colorscale='Plasma')
            ))
        
        fig.update_layout(
            title='3D Sensor Trajectories',
            scene=dict(
                xaxis_title='X-axis',
                yaxis_title='Y-axis',
                zaxis_title='Z-axis'
            ),
            height=800
        )
        
        return fig
    
    def plot_comprehensive_spectrogram(self, data: pd.DataFrame, 
                                     sensor_col: str) -> plt.Figure:
        """Create comprehensive spectrogram with multiple views"""
        fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
        
        if sensor_col not in data.columns:
            return fig
        
        signal_data = data[sensor_col].values
        
        # Remove NaN
        signal_data = signal_data[~np.isnan(signal_data)]
        
        if len(signal_data) < 256:
            return fig
        
        # Time series
        ax1 = plt.subplot(3, 1, 1)
        time = np.arange(len(signal_data)) / self.sampling_rate
        ax1.plot(time, signal_data, 'b-', linewidth=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{sensor_col.upper()} Signal Analysis')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        ax2 = plt.subplot(3, 1, 2)
        f, t, Sxx = signal.spectrogram(signal_data, fs=self.sampling_rate, 
                                      nperseg=256, noverlap=192)
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        im = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='jet')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim(0, 50)
        plt.colorbar(im, ax=ax2, label='Power (dB)')
        
        # Power Spectral Density
        ax3 = plt.subplot(3, 1, 3)
        frequencies, psd = welch(signal_data, fs=self.sampling_rate, nperseg=512)
        ax3.semilogy(frequencies, psd, 'g-')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('PSD (units²/Hz)')
        ax3.set_xlim(0, 50)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_autocorrelation_analysis(self, df: pd.DataFrame, 
                                    max_lag: int = 200) -> plt.Figure:
        """Plot autocorrelation for all sensors"""
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        n_sensors = len(available_sensors)
        fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 2*n_sensors), 
                                dpi=self.dpi, sharex=True)
        
        if n_sensors == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(available_sensors):
            data = df[sensor].dropna().values
            
            if len(data) > max_lag:
                # Calculate autocorrelation
                autocorr = [1.0]
                for lag in range(1, max_lag):
                    if lag < len(data):
                        corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                        autocorr.append(corr)
                    else:
                        autocorr.append(0)
                
                # Plot
                axes[idx].stem(range(max_lag), autocorr, basefmt=' ')
                axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[idx].axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
                axes[idx].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
                axes[idx].set_ylabel(f'{sensor.upper()}')
                axes[idx].set_ylim(-1, 1)
                axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Lag (samples)')
        fig.suptitle('Autocorrelation Analysis', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_sensor_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive correlation heatmap"""
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        if len(available_sensors) < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = df[available_sensors].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Sensor Cross-Correlation Matrix',
            width=600,
            height=600
        )
        
        return fig
    
    def plot_movement_patterns(self, df: pd.DataFrame, 
                              window_size: int = 50) -> plt.Figure:
        """Visualize movement patterns and transitions"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=self.dpi, sharex=True)
        
        # Calculate movement intensity
        if 'acc_mag' in df.columns:
            # Movement intensity
            movement_intensity = df['acc_mag'].rolling(window=window_size).std()
            axes[0].plot(movement_intensity.index, movement_intensity.values, 'b-')
            axes[0].fill_between(movement_intensity.index, movement_intensity.values, 
                               alpha=0.3)
            axes[0].set_ylabel('Movement Intensity')
            axes[0].set_title('Movement Pattern Analysis')
            axes[0].grid(True, alpha=0.3)
        
        # Rotation intensity
        if 'gyro_mag' in df.columns:
            rotation_intensity = df['gyro_mag'].rolling(window=window_size).std()
            axes[1].plot(rotation_intensity.index, rotation_intensity.values, 'r-')
            axes[1].fill_between(rotation_intensity.index, rotation_intensity.values, 
                               alpha=0.3, color='red')
            axes[1].set_ylabel('Rotation Intensity')
            axes[1].grid(True, alpha=0.3)
        
        # Combined activity score
        if 'acc_mag' in df.columns and 'gyro_mag' in df.columns:
            # Normalize and combine
            acc_norm = (df['acc_mag'] - df['acc_mag'].mean()) / df['acc_mag'].std()
            gyro_norm = (df['gyro_mag'] - df['gyro_mag'].mean()) / df['gyro_mag'].std()
            activity_score = np.sqrt(acc_norm**2 + gyro_norm**2)
            
            axes[2].plot(activity_score.index, activity_score.values, 'g-', alpha=0.7)
            axes[2].set_ylabel('Activity Score')
            axes[2].set_xlabel('Sample')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig