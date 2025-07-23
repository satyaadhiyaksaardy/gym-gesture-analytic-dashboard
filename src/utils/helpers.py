"""
Helper utility functions
"""

import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
from typing import Union, Optional


def safe_plot_close(fig: plt.Figure) -> None:
    """Safely close matplotlib figure to prevent memory leaks"""
    try:
        plt.close(fig)
        gc.collect()
    except:
        pass


def safe_division(numerator: float, denominator: float, default: float = 0) -> float:
    """Safe division that handles zero denominator"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def format_metric(value: Union[int, float], metric_type: str = 'general') -> str:
    """Format metric values for display"""
    if pd.isna(value):
        return 'N/A'
    
    if metric_type == 'count':
        return f"{int(value):,}"
    elif metric_type == 'percentage':
        return f"{value:.2f}%"
    elif metric_type == 'time_ms':
        return f"{value:.2f}ms"
    elif metric_type == 'memory_mb':
        return f"{value:.2f} MB"
    elif metric_type == 'score':
        return f"{value:.3f}"
    else:
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:.2f}"


def validate_numeric_data(data: np.ndarray) -> bool:
    """Validate that data contains valid numeric values"""
    if len(data) == 0:
        return False
    
    # Check for all NaN
    if np.all(np.isnan(data)):
        return False
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        return False
    
    return True


def calculate_percentile_ranges(data: pd.Series, percentiles: list = [25, 50, 75]) -> dict:
    """Calculate percentile ranges for data"""
    ranges = {}
    for p in percentiles:
        ranges[f'p{p}'] = data.quantile(p / 100)
    return ranges


def get_memory_usage(df: pd.DataFrame) -> dict:
    """Get detailed memory usage statistics"""
    total_memory = df.memory_usage(deep=True).sum()
    
    memory_stats = {
        'total_mb': total_memory / 1024**2,
        'per_column': {},
        'dtype_summary': {}
    }
    
    # Per column memory
    for col in df.columns:
        memory_stats['per_column'][col] = df[col].memory_usage(deep=True) / 1024**2
    
    # By dtype
    for dtype in df.dtypes.unique():
        cols_with_dtype = df.select_dtypes(include=[dtype]).columns
        dtype_memory = sum(df[col].memory_usage(deep=True) for col in cols_with_dtype)
        memory_stats['dtype_summary'][str(dtype)] = dtype_memory / 1024**2
    
    return memory_stats


def remove_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return data[(data >= lower_bound) & (data <= upper_bound)]


def create_time_windows(df: pd.DataFrame, window_size_ms: float, 
                       timestamp_col: str = 'timestamp') -> list:
    """Create time windows for analysis"""
    if timestamp_col not in df.columns:
        return []
    
    windows = []
    start_time = df[timestamp_col].min()
    end_time = df[timestamp_col].max()
    
    current_start = start_time
    while current_start < end_time:
        current_end = current_start + window_size_ms
        window_data = df[(df[timestamp_col] >= current_start) & 
                        (df[timestamp_col] < current_end)]
        
        if len(window_data) > 0:
            windows.append({
                'start_time': current_start,
                'end_time': current_end,
                'data': window_data,
                'sample_count': len(window_data)
            })
        
        current_start = current_end
    
    return windows


def interpolate_missing_data(data: pd.Series, method: str = 'linear', 
                           limit: Optional[int] = None) -> pd.Series:
    """Interpolate missing values in time series data"""
    if method == 'linear':
        return data.interpolate(method='linear', limit=limit)
    elif method == 'polynomial':
        return data.interpolate(method='polynomial', order=2, limit=limit)
    elif method == 'spline':
        return data.interpolate(method='spline', order=3, limit=limit)
    else:
        return data.interpolate(method='linear', limit=limit)


def detect_signal_quality(data: pd.Series) -> dict:
    """Assess signal quality metrics"""
    quality_metrics = {
        'completeness': (data.notna().sum() / len(data)) * 100,
        'snr_estimate': 0,
        'stability': 0,
        'quality_score': 0
    }
    
    # Estimate SNR (simplified)
    if len(data.dropna()) > 10:
        signal_power = np.var(data.dropna())
        noise_estimate = np.var(np.diff(data.dropna()))
        quality_metrics['snr_estimate'] = 10 * np.log10(
            signal_power / noise_estimate) if noise_estimate > 0 else 0
    
    # Stability (coefficient of variation)
    if data.mean() != 0:
        quality_metrics['stability'] = (data.std() / data.mean()) * 100
    
    # Overall quality score
    completeness_score = quality_metrics['completeness']
    snr_score = min(100, max(0, quality_metrics['snr_estimate'] * 10))
    stability_score = max(0, 100 - quality_metrics['stability'])
    
    quality_metrics['quality_score'] = (
        completeness_score * 0.4 + 
        snr_score * 0.3 + 
        stability_score * 0.3
    )
    
    return quality_metrics


def generate_summary_statistics(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Generate comprehensive summary statistics"""
    summary_data = []
    
    for col in columns:
        if col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                summary_data.append({
                    'column': col,
                    'count': len(col_data),
                    'missing': df[col].isna().sum(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'p25': col_data.quantile(0.25),
                    'median': col_data.median(),
                    'p75': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skew': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
    
    return pd.DataFrame(summary_data)


def create_exercise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary by exercise type"""
    if 'exercise_type' not in df.columns:
        return pd.DataFrame()
    
    summary = []
    for exercise in df['exercise_type'].unique():
        exercise_data = df[df['exercise_type'] == exercise]
        
        summary.append({
            'exercise_type': exercise,
            'total_samples': len(exercise_data),
            'unique_athletes': exercise_data['athlete_id'].nunique() if 'athlete_id' in df.columns else 0,
            'unique_sets': len(exercise_data.groupby(['athlete_id', 'set_number'])) if all(col in df.columns for col in ['athlete_id', 'set_number']) else 0,
            'avg_duration_ms': exercise_data.groupby(['athlete_id', 'set_number', 'rep_number'])['timestamp'].apply(lambda x: x.max() - x.min()).mean() if 'timestamp' in df.columns else 0,
            'sample_rate_hz': 1000 / exercise_data['timestamp'].diff().mean() if 'timestamp' in df.columns else 0
        })
    
    return pd.DataFrame(summary)


def check_data_consistency(df: pd.DataFrame) -> dict:
    """Check data consistency across different dimensions"""
    issues = {
        'duplicate_rows': 0,
        'missing_sequences': [],
        'inconsistent_sampling': [],
        'data_gaps': []
    }
    
    # Check for duplicates
    issues['duplicate_rows'] = df.duplicated().sum()
    
    # Check for missing sequences in rep numbers
    if all(col in df.columns for col in ['athlete_id', 'exercise_type', 'set_number', 'rep_number']):
        groups = df.groupby(['athlete_id', 'exercise_type', 'set_number'])
        
        for name, group in groups:
            rep_numbers = sorted(group['rep_number'].unique())
            expected_reps = list(range(min(rep_numbers), max(rep_numbers) + 1))
            missing_reps = set(expected_reps) - set(rep_numbers)
            
            if missing_reps:
                issues['missing_sequences'].append({
                    'athlete_id': name[0],
                    'exercise_type': name[1],
                    'set_number': name[2],
                    'missing_reps': list(missing_reps)
                })
    
    return issues