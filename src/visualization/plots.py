"""
Basic plotting functions for data visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from config import PLOT_STYLE, FIGURE_DPI, COLOR_PALETTE
from utils.helpers import safe_plot_close


class PlotGenerator:
    """Class for generating basic plots"""
    
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.dpi = FIGURE_DPI
        self.palette = COLOR_PALETTE
    
    def plot_outliers_by_sensor(self, outlier_counts: dict) -> plt.Figure:
        """Create bar plot of outliers by sensor"""
        fig, ax = plt.subplots(figsize=(8, 4), dpi=self.dpi)
        
        if outlier_counts:
            sensors = list(outlier_counts.keys())
            counts = list(outlier_counts.values())
            
            ax.bar(sensors, counts, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Sensor')
            ax.set_ylabel('Outlier Count')
            ax.set_title('Outliers by Sensor Channel')
            
            # Add value labels on bars
            for i, (sensor, count) in enumerate(zip(sensors, counts)):
                ax.text(i, count + max(counts) * 0.01, str(count), 
                       ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        return fig
    
    def plot_duration_histogram(self, durations: pd.Series, 
                              remove_outliers: bool = True) -> plt.Figure:
        """Create histogram of repetition durations"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        data = durations.copy()
        
        if remove_outliers and len(data) > 5:
            # Remove outliers for better visualization
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.2f}ms')
            ax.set_xlabel('Duration (ms)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Repetition Durations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               columns: Optional[List[str]] = None) -> plt.Figure:
        """Create correlation heatmap"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        corr_matrix = data[columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def plot_time_series(self, data: pd.DataFrame, columns: List[str],
                        title: str = "Time Series Plot") -> plt.Figure:
        """Create time series plot for multiple columns"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        x = range(len(data))
        
        for i, col in enumerate(columns):
            if col in data.columns:
                ax.plot(x, data[col].values, label=col, alpha=0.8)
        
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution_comparison(self, data1: pd.Series, data2: pd.Series,
                                   label1: str = "Dataset 1", 
                                   label2: str = "Dataset 2",
                                   title: str = "Distribution Comparison") -> plt.Figure:
        """Compare distributions of two datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # Histograms
        ax1.hist(data1, bins=30, alpha=0.5, label=label1, color='blue', density=True)
        ax1.hist(data2, bins=30, alpha=0.5, label=label2, color='red', density=True)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution Overlay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plots
        ax2.boxplot([data1, data2], labels=[label1, label2])
        ax2.set_ylabel('Value')
        ax2.set_title('Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_sampling_intervals(self, intervals: pd.Series, 
                               expected_interval: float = 10) -> plt.Figure:
        """Plot sampling interval analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=self.dpi)
        
        # Time series of intervals
        ax1.plot(range(len(intervals)), intervals.values, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(y=expected_interval, color='green', linestyle='--', 
                   label=f'Expected ({expected_interval}ms)')
        ax1.axhline(y=intervals.mean(), color='red', linestyle='--', 
                   label=f'Mean ({intervals.mean():.1f}ms)')
        
        # Add standard deviation band
        ax1.fill_between(range(len(intervals)), 
                        intervals.mean() - intervals.std(),
                        intervals.mean() + intervals.std(),
                        alpha=0.3, color='red', label='±1 STD')
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Interval (ms)')
        ax1.set_title('Sampling Intervals Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of intervals
        ax2.hist(intervals, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=expected_interval, color='green', linestyle='--', linewidth=2)
        ax2.axvline(x=intervals.mean(), color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Interval (ms)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Sampling Intervals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_exercise_distribution(self, exercise_counts: pd.Series) -> plt.Figure:
        """Plot distribution of samples by exercise type"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        exercise_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Exercise Type')
        ax.set_ylabel('Sample Count')
        ax.set_title('Distribution of Samples by Exercise Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(exercise_counts.values):
            ax.text(i, v + max(exercise_counts) * 0.01, f'{v:,}', 
                   ha='center', va='bottom')
        
        # Add percentage labels
        total = exercise_counts.sum()
        for i, v in enumerate(exercise_counts.values):
            pct = v / total * 100
            ax.text(i, v / 2, f'{pct:.1f}%', ha='center', va='center', 
                   color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20) -> plt.Figure:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Select top N features
        plot_df = importance_df.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df['cv'], color='green', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['feature'])
        ax.set_xlabel('Coefficient of Variation')
        ax.set_title(f'Top {top_n} Features by Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig