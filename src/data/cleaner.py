"""
Data cleaning and outlier detection module
"""

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from typing import Tuple, Optional
from config import OUTLIER_Z_SCORE_THRESHOLD, REQUIRED_SENSOR_COLS


class DataCleaner:
    """Class for data cleaning and outlier detection"""
    
    def __init__(self, z_score_threshold: float = OUTLIER_Z_SCORE_THRESHOLD):
        self.z_score_threshold = z_score_threshold
    
    @st.cache_data
    def compute_magnitudes(_self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute accelerometer and gyroscope magnitudes
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added magnitude columns
        """
        df = df.copy()
        
        # Accelerometer magnitude
        acc_cols = ['ax', 'ay', 'az']
        if all(col in df.columns for col in acc_cols):
            # Replace NaN with 0 before computation
            for col in acc_cols:
                df[col] = df[col].fillna(0)
            
            df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
            # Handle any infinite values
            df['acc_mag'] = df['acc_mag'].replace([np.inf, -np.inf], np.nan)
        
        # Gyroscope magnitude
        gyro_cols = ['gx', 'gy', 'gz']
        if all(col in df.columns for col in gyro_cols):
            # Replace NaN with 0 before computation
            for col in gyro_cols:
                df[col] = df[col].fillna(0)
            
            df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
            # Handle any infinite values
            df['gyro_mag'] = df['gyro_mag'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    @st.cache_data
    def detect_outliers(_self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Detect outliers using z-score method
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold (uses instance default if None)
            
        Returns:
            Boolean Series indicating outliers
        """
        if threshold is None:
            threshold = _self.z_score_threshold
            
        df = df.copy()
        sensor_cols = REQUIRED_SENSOR_COLS + ['acc_mag', 'gyro_mag']
        
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in sensor_cols:
            if col in df.columns:
                # Remove NaN values before computing z-scores
                col_data = df[col].dropna()
                if len(col_data) > 0 and col_data.std() > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    # Map back to original indices
                    outlier_indices = col_data[z_scores > threshold].index
                    outlier_mask.loc[outlier_indices] = True
        
        return outlier_mask
    
    def get_outlier_statistics(self, df: pd.DataFrame, outlier_mask: pd.Series) -> dict:
        """Get detailed outlier statistics"""
        outlier_stats = {
            'total_outliers': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(df) * 100),
            'outliers_by_sensor': {}
        }
        
        # Outliers by sensor
        for col in REQUIRED_SENSOR_COLS:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0 and col_data.std() > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    outlier_stats['outliers_by_sensor'][col] = np.sum(z_scores > self.z_score_threshold)
        
        return outlier_stats
    
    def clean_data(self, df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
        """Remove outliers from the dataset"""
        return df[~outlier_mask].copy()
    
    def interpolate_missing_values(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate missing values in sensor data
        
        Args:
            df: Input DataFrame
            method: Interpolation method ('linear', 'polynomial', 'spline')
            
        Returns:
            DataFrame with interpolated values
        """
        df = df.copy()
        
        # Group by repetition for interpolation
        grouping_cols = ['athlete_id', 'exercise_type', 'weight_kg', 'set_number', 'rep_number']
        available_grouping_cols = [col for col in grouping_cols if col in df.columns]
        
        if available_grouping_cols:
            # Interpolate within each group
            sensor_cols = [col for col in REQUIRED_SENSOR_COLS if col in df.columns]
            
            for col in sensor_cols:
                df[col] = df.groupby(available_grouping_cols)[col].transform(
                    lambda x: x.interpolate(method=method, limit_direction='both')
                )
        
        return df
    
    def remove_duplicate_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows from the dataset
        
        Returns:
            Tuple of (cleaned DataFrame, number of duplicates removed)
        """
        initial_len = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_len - len(df_cleaned)
        
        return df_cleaned, duplicates_removed
    
    def handle_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing timestamps by generating synthetic timestamps
        based on expected sampling rate
        """
        if 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        # Group by repetition
        grouping_cols = ['athlete_id', 'exercise_type', 'weight_kg', 'set_number', 'rep_number']
        available_grouping_cols = [col for col in grouping_cols if col in df.columns]
        
        if available_grouping_cols:
            def fill_timestamps(group):
                if group['timestamp'].notna().sum() >= 2:
                    # Interpolate missing timestamps
                    group['timestamp'] = group['timestamp'].interpolate(method='linear')
                return group
            
            df = df.groupby(available_grouping_cols).apply(fill_timestamps)
        
        return df