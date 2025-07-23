"""
Signal visualization module for sensor data
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy import signal
import streamlit as st
from typing import Optional, Tuple, List
from config import PLOT_STYLE, FIGURE_DPI, SPECTROGRAM_NPERSEG, SAMPLING_RATE_HZ
from utils.helpers import safe_plot_close


class SignalVisualizer:
    """Class for visualizing sensor signals"""
    
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.dpi = FIGURE_DPI
        self.sampling_rate = SAMPLING_RATE_HZ
    
    def plot_sensor_signals(self, data: pd.DataFrame, 
                           sensor_type: str = 'accelerometer') -> plt.Figure:
        """Plot accelerometer or gyroscope signals"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        if sensor_type == 'accelerometer':
            cols = ['ax', 'ay', 'az']
            ylabel = 'Acceleration (g)'
            title = 'Accelerometer Data'
        else:
            cols = ['gx', 'gy', 'gz']
            ylabel = 'Angular Velocity (deg/s)'
            title = 'Gyroscope Data'
        
        available_cols = [col for col in cols if col in data.columns]
        
        if available_cols:
            samples = range(len(data))
            
            for col in available_cols:
                label = f'{col.upper()}-axis'
                ax.plot(samples, data[col].values, label=label, alpha=0.8)
            
            ax.set_xlabel('Sample')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_magnitude_signals(self, data: pd.DataFrame) -> plt.Figure:
        """Plot magnitude signals for accelerometer and gyroscope"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.dpi, sharex=True)
        
        samples = range(len(data))
        
        # Accelerometer magnitude
        if 'acc_mag' in data.columns:
            ax1.plot(samples, data['acc_mag'].values, 'b-', alpha=0.8)
            ax1.set_ylabel('Acceleration Magnitude (g)')
            ax1.set_title('Accelerometer Magnitude')
            ax1.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = data['acc_mag'].mean()
            ax1.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5,
                       label=f'Mean: {mean_val:.2f}')
            ax1.legend()
        
        # Gyroscope magnitude
        if 'gyro_mag' in data.columns:
            ax2.plot(samples, data['gyro_mag'].values, 'g-', alpha=0.8)
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Angular Velocity Magnitude (deg/s)')
            ax2.set_title('Gyroscope Magnitude')
            ax2.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = data['gyro_mag'].mean()
            ax2.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5,
                       label=f'Mean: {mean_val:.2f}')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(self, data: np.ndarray, title: str = "Spectrogram") -> plt.Figure:
        """Create spectrogram of signal"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) > 20:
            # Adjust nperseg based on signal length
            nperseg = min(SPECTROGRAM_NPERSEG, len(data_clean) // 4, len(data_clean) - 1)
            
            if nperseg > 1:
                f, t, Sxx = signal.spectrogram(data_clean, fs=self.sampling_rate, nperseg=nperseg)
                
                # Convert to dB
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                
                im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [sec]')
                ax.set_title(title)
                ax.set_ylim(0, 50)  # Focus on relevant frequency range
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Power [dB]')
        
        plt.tight_layout()
        return fig
    
    def plot_phase_space(self, data1: pd.Series, data2: pd.Series,
                        label1: str = "Signal 1", label2: str = "Signal 2") -> plt.Figure:
        """Create phase space plot"""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        
        # Clean data
        data1_clean = data1.dropna()
        data2_clean = data2.dropna()
        
        if len(data1_clean) > 1 and len(data2_clean) > 1:
            min_len = min(len(data1_clean) - 1, len(data2_clean) - 1)
            
            if min_len > 0:
                scatter = ax.scatter(data1_clean.values[:min_len], 
                                   data2_clean.values[1:min_len+1],
                                   c=range(min_len), cmap='viridis',
                                   alpha=0.6, s=30)
                
                ax.set_xlabel(f'{label1} (t)')
                ax.set_ylabel(f'{label2} (t+1)')
                ax.set_title('Phase Space Plot')
                ax.grid(True, alpha=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Time')
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_statistics(self, data: pd.Series, window: int = 20) -> plt.Figure:
        """Plot signal with rolling mean and standard deviation"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        samples = range(len(data))
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = data.rolling(window=window, center=True, min_periods=1).std()
        
        # Plot raw signal
        ax.plot(samples, data.values, alpha=0.3, label='Raw', color='gray')
        
        # Plot rolling mean
        ax.plot(samples, rolling_mean.values, label=f'Rolling Mean (w={window})',
               linewidth=2, color='blue')
        
        # Plot confidence interval
        valid_idx = ~rolling_std.isna()
        if valid_idx.sum() > 0:
            ax.fill_between(np.array(samples)[valid_idx],
                          (rolling_mean - rolling_std).values[valid_idx],
                          (rolling_mean + rolling_std).values[valid_idx],
                          alpha=0.2, label='±1 std dev', color='blue')
        
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title('Signal with Rolling Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectory(self, data: pd.DataFrame, cols: List[str]) -> plt.Figure:
        """Plot 3D trajectory of sensor data"""
        fig = plt.figure(figsize=(10, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        if all(col in data.columns for col in cols) and len(cols) >= 3:
            x = data[cols[0]].values
            y = data[cols[1]].values
            z = data[cols[2]].values
            
            # Create color map for time
            colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
            
            # Plot trajectory
            for i in range(len(data) - 1):
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2)
            
            # Add start and end markers
            ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start')
            ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='s', label='End')
            
            ax.set_xlabel(cols[0].upper())
            ax.set_ylabel(cols[1].upper())
            ax.set_zlabel(cols[2].upper())
            ax.set_title('3D Sensor Trajectory')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_spectrum(self, data: np.ndarray, title: str = "Frequency Spectrum") -> plt.Figure:
        """Plot frequency spectrum of signal"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) > 10:
            # Compute FFT
            fft_vals = np.fft.fft(data_clean)
            freqs = np.fft.fftfreq(len(data_clean), 1/self.sampling_rate)
            
            # Get positive frequencies only
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = np.abs(fft_vals[pos_mask])
            
            # Plot spectrum
            ax.plot(freqs, fft_vals, 'b-', linewidth=1.5)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)  # Focus on relevant frequency range
            
            # Mark dominant frequency
            if len(fft_vals) > 0:
                dom_idx = np.argmax(fft_vals)
                ax.plot(freqs[dom_idx], fft_vals[dom_idx], 'ro', markersize=8)
                ax.annotate(f'{freqs[dom_idx]:.1f} Hz', 
                          xy=(freqs[dom_idx], fft_vals[dom_idx]),
                          xytext=(freqs[dom_idx] + 2, fft_vals[dom_idx]),
                          fontsize=10)
        
        plt.tight_layout()
        return fig