"""
Timing and sampling analysis module
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional
from config import (
    SAMPLING_INTERVAL_MS, IRREGULAR_SAMPLING_THRESHOLD_MS,
    MAX_REP_DURATION_MS, SAMPLING_CONSISTENCY_THRESHOLD_MS,
    METADATA_COLS
)


class TimingAnalyzer:
    """Class for analyzing timing and sampling patterns in sensor data"""
    
    def __init__(self):
        self.expected_interval = SAMPLING_INTERVAL_MS
        self.irregular_threshold = IRREGULAR_SAMPLING_THRESHOLD_MS
        self.max_duration = MAX_REP_DURATION_MS
    
    @st.cache_data
    def analyze_sampling_consistency(_self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sampling intervals by group
        
        Args:
            df: Input DataFrame with timestamp column
            
        Returns:
            DataFrame with sampling statistics for each group
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        results = []
        
        # Check if required columns exist
        if not all(col in df.columns for col in METADATA_COLS):
            return pd.DataFrame()
        
        groups = df.groupby(METADATA_COLS)
        
        for name, group in groups:
            if len(group) > 1:  # Need at least 2 points for intervals
                grp = group.sort_values('timestamp')
                diffs = grp['timestamp'].diff().dropna()
                
                if len(diffs) > 0:
                    results.append({
                        'athlete_id': name[0],
                        'exercise_type': name[1],
                        'weight_kg': name[2],
                        'set_number': name[3],
                        'rep_number': name[4],
                        'mean_interval_ms': diffs.mean(),
                        'std_interval_ms': diffs.std() if len(diffs) > 1 else 0,
                        'min_interval_ms': diffs.min(),
                        'max_interval_ms': diffs.max(),
                        'cv_interval': diffs.std() / diffs.mean() if diffs.mean() > 0 else 0,
                        'sample_count': len(group),
                        'duration_ms': grp['timestamp'].max() - grp['timestamp'].min()
                    })
        
        return pd.DataFrame(results)
    
    @st.cache_data
    def compute_rep_durations(_self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute duration for each repetition
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with repetition durations
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        results = []
        
        if not all(col in df.columns for col in METADATA_COLS):
            return pd.DataFrame()
        
        groups = df.groupby(METADATA_COLS)
        
        for name, grp in groups:
            if len(grp) > 1:
                # Calculate duration
                duration = grp['timestamp'].max() - grp['timestamp'].min()
                
                # Filter out extreme durations
                if duration <= _self.max_duration:
                    results.append({
                        'athlete_id': name[0],
                        'exercise_type': name[1],
                        'weight_kg': name[2],
                        'set_number': name[3],
                        'rep_number': name[4],
                        'duration_ms': duration,
                        'sample_count': len(grp),
                        'samples_per_second': len(grp) / (duration / 1000) if duration > 0 else 0
                    })
        
        return pd.DataFrame(results)
    
    def identify_irregular_sampling(self, sampling_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify groups with irregular sampling patterns
        
        Args:
            sampling_df: DataFrame from analyze_sampling_consistency
            
        Returns:
            DataFrame with irregular sampling groups
        """
        if sampling_df.empty:
            return pd.DataFrame()
        
        # Filter groups with high standard deviation
        irregular_groups = sampling_df[
            sampling_df['std_interval_ms'] > self.irregular_threshold
        ].copy()
        
        # Add severity score
        if not irregular_groups.empty:
            irregular_groups['severity_score'] = (
                irregular_groups['std_interval_ms'] / self.expected_interval
            )
            
            # Sort by severity
            irregular_groups = irregular_groups.sort_values('severity_score', ascending=False)
        
        return irregular_groups
    
    def analyze_timing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive timing pattern analysis
        
        Returns:
            Dictionary with timing analysis results
        """
        results = {
            'has_timestamp': 'timestamp' in df.columns,
            'total_duration_ms': 0,
            'overall_sampling_rate': 0,
            'irregular_sampling_rate': 0,
            'timing_quality': 'Unknown'
        }
        
        if not results['has_timestamp']:
            return results
        
        # Calculate overall duration
        results['total_duration_ms'] = df['timestamp'].max() - df['timestamp'].min()
        
        # Calculate overall sampling rate
        if results['total_duration_ms'] > 0:
            results['overall_sampling_rate'] = (
                len(df) / (results['total_duration_ms'] / 1000)
            )
        
        # Analyze sampling consistency
        sampling_df = self.analyze_sampling_consistency(df)
        if not sampling_df.empty:
            # Calculate irregular sampling rate
            irregular_count = (sampling_df['std_interval_ms'] > self.irregular_threshold).sum()
            results['irregular_sampling_rate'] = irregular_count / len(sampling_df) * 100
            
            # Determine timing quality
            mean_std = sampling_df['std_interval_ms'].mean()
            if mean_std < 5:
                results['timing_quality'] = 'Excellent'
            elif mean_std < 15:
                results['timing_quality'] = 'Good'
            elif mean_std < 30:
                results['timing_quality'] = 'Fair'
            else:
                results['timing_quality'] = 'Poor'
            
            # Add detailed statistics
            results['mean_interval_ms'] = sampling_df['mean_interval_ms'].mean()
            results['std_interval_ms'] = sampling_df['std_interval_ms'].mean()
            results['min_interval_ms'] = sampling_df['min_interval_ms'].min()
            results['max_interval_ms'] = sampling_df['max_interval_ms'].max()
        
        return results
    
    def get_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze temporal patterns across repetitions
        
        Returns:
            DataFrame with temporal pattern analysis
        """
        if 'timestamp' not in df.columns or 'rep_number' not in df.columns:
            return pd.DataFrame()
        
        # Analyze patterns by rep number
        patterns = []
        
        groups = df.groupby('rep_number')
        for rep_num, group in groups:
            if len(group) > 0:
                patterns.append({
                    'rep_number': rep_num,
                    'sample_count': len(group),
                    'mean_timestamp': group['timestamp'].mean(),
                    'timestamp_range': group['timestamp'].max() - group['timestamp'].min()
                })
        
        return pd.DataFrame(patterns)
    
    def detect_data_gaps(self, df: pd.DataFrame, gap_threshold_ms: float = 50) -> pd.DataFrame:
        """
        Detect gaps in data collection
        
        Args:
            df: Input DataFrame
            gap_threshold_ms: Threshold for detecting gaps
            
        Returns:
            DataFrame with detected gaps
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        gaps = []
        
        # Check gaps within each group
        if all(col in df.columns for col in METADATA_COLS):
            groups = df.groupby(METADATA_COLS)
            
            for name, group in groups:
                if len(group) > 1:
                    grp = group.sort_values('timestamp')
                    diffs = grp['timestamp'].diff()
                    
                    # Find gaps
                    gap_indices = diffs[diffs > gap_threshold_ms].index
                    
                    for idx in gap_indices:
                        gap_size = diffs.loc[idx]
                        gaps.append({
                            'athlete_id': name[0],
                            'exercise_type': name[1],
                            'weight_kg': name[2],
                            'set_number': name[3],
                            'rep_number': name[4],
                            'gap_start_time': grp.loc[idx-1, 'timestamp'] if idx > 0 else 0,
                            'gap_end_time': grp.loc[idx, 'timestamp'],
                            'gap_duration_ms': gap_size,
                            'gap_severity': 'Large' if gap_size > 100 else 'Medium' if gap_size > 50 else 'Small'
                        })
        
        return pd.DataFrame(gaps)
    
    def calculate_sampling_metrics(self, sampling_df: pd.DataFrame) -> Dict:
        """Calculate summary metrics for sampling quality"""
        if sampling_df.empty:
            return {}
        
        metrics = {
            'total_groups': len(sampling_df),
            'groups_with_good_sampling': (sampling_df['std_interval_ms'] < 5).sum(),
            'groups_with_irregular_sampling': (sampling_df['std_interval_ms'] > self.irregular_threshold).sum(),
            'overall_mean_interval': sampling_df['mean_interval_ms'].mean(),
            'overall_std_interval': sampling_df['std_interval_ms'].mean(),
            'worst_std_interval': sampling_df['std_interval_ms'].max(),
            'best_std_interval': sampling_df['std_interval_ms'].min(),
            'sampling_quality_score': 0
        }
        
        # Calculate quality score (0-100)
        if metrics['overall_mean_interval'] > 0:
            deviation_from_expected = abs(metrics['overall_mean_interval'] - self.expected_interval)
            consistency_score = max(0, 100 - (metrics['overall_std_interval'] / self.expected_interval * 100))
            accuracy_score = max(0, 100 - (deviation_from_expected / self.expected_interval * 100))
            
            metrics['sampling_quality_score'] = (consistency_score + accuracy_score) / 2
        
        return metrics