"""
Data loading and validation module
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from config import (
    MAX_FILE_SIZE_MB, REQUIRED_SENSOR_COLS, METADATA_COLS,
    OPTIONAL_COLS
)


class DataLoader:
    """Class for loading and validating sensor data"""
    
    def __init__(self):
        self.required_cols = REQUIRED_SENSOR_COLS + METADATA_COLS
        self.numeric_cols = REQUIRED_SENSOR_COLS + ['weight_kg', 'set_number', 'rep_number', 'timestamp']
    
    @st.cache_data
    def load_data(_self, uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load CSV data with comprehensive error handling
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (DataFrame or None, error message or None)
        """
        try:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return None, f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB"
            
            # Read CSV with error handling
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Basic validation
            if df.empty:
                return None, "The uploaded file is empty"
            
            # Clean numeric columns
            df = _self._clean_numeric_columns(df)
            
            # Remove rows with all NaN sensor values
            df = _self._remove_empty_sensor_rows(df)
            
            # Validate data structure
            is_valid, message = _self.validate_dataframe(df)
            if not is_valid:
                return None, message
            
            return df, None
            
        except pd.errors.EmptyDataError:
            return None, "The uploaded file is empty or corrupted"
        except pd.errors.ParserError as e:
            return None, f"Error parsing CSV: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error loading file: {str(e)}"
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns by converting to numeric and handling errors"""
        df = df.copy()
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _remove_empty_sensor_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where all sensor values are NaN"""
        sensor_cols_present = [col for col in REQUIRED_SENSOR_COLS if col in df.columns]
        if sensor_cols_present:
            df = df.dropna(subset=sensor_cols_present, how='all')
        return df
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the dataframe structure
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty or None"
        
        # Check for required columns
        missing_cols = []
        for col in self.required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check if we have enough data
        if len(df) < 10:
            return False, "Insufficient data points (minimum 10 required)"
        
        # Check if sensor columns have valid numeric data
        sensor_data_check = []
        for col in REQUIRED_SENSOR_COLS:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                sensor_data_check.append(non_null_count > 0)
        
        if not any(sensor_data_check):
            return False, "No valid sensor data found"
        
        return True, "Valid"
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the loaded data"""
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'has_timestamp': 'timestamp' in df.columns,
            'sensor_columns': [col for col in REQUIRED_SENSOR_COLS if col in df.columns],
            'metadata_columns': [col for col in METADATA_COLS if col in df.columns],
            'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        }
        
        # Add unique counts for metadata
        for col in ['athlete_id', 'exercise_type']:
            if col in df.columns:
                info[f'unique_{col}'] = df[col].nunique()
        
        return info