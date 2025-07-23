"""
Statistical analysis module
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import streamlit as st


class StatisticalAnalyzer:
    """Class for performing statistical analysis on sensor data"""
    
    def __init__(self):
        pass
    
    def calculate_basic_statistics(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Calculate basic statistics for specified columns"""
        stats_data = []
        
        for col in columns:
            if col in df.columns:
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    stats_data.append({
                        'column': col,
                        'count': len(col_data),
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'p25': col_data.quantile(0.25),
                        'median': col_data.median(),
                        'p75': col_data.quantile(0.75),
                        'max': col_data.max(),
                        'range': col_data.max() - col_data.min(),
                        'cv': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0,
                        'skew': col_data.skew(),
                        'kurtosis': col_data.kurtosis()
                    })
        
        return pd.DataFrame(stats_data)
    
    @st.cache_data
    def compare_distributions(_self, df: pd.DataFrame, groupby_col: str, 
                            target_col: str) -> Dict:
        """Compare distributions across groups"""
        if groupby_col not in df.columns or target_col not in df.columns:
            return {}
        
        results = {
            'groups': {},
            'statistical_tests': {},
            'summary': {}
        }
        
        # Group analysis
        groups = df.groupby(groupby_col)[target_col]
        
        for name, group_data in groups:
            clean_data = group_data.dropna()
            if len(clean_data) > 0:
                results['groups'][name] = {
                    'count': len(clean_data),
                    'mean': clean_data.mean(),
                    'std': clean_data.std(),
                    'median': clean_data.median(),
                    'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25)
                }
        
        # Statistical tests (if more than one group)
        group_names = list(results['groups'].keys())
        if len(group_names) >= 2:
            # Prepare data for tests
            group_data_list = [df[df[groupby_col] == name][target_col].dropna() 
                              for name in group_names]
            
            # ANOVA test
            if all(len(data) > 0 for data in group_data_list):
                f_stat, p_value = stats.f_oneway(*group_data_list)
                results['statistical_tests']['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            # Kruskal-Wallis test (non-parametric)
            if all(len(data) > 0 for data in group_data_list):
                h_stat, p_value = stats.kruskal(*group_data_list)
                results['statistical_tests']['kruskal_wallis'] = {
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Summary statistics
        all_means = [v['mean'] for v in results['groups'].values()]
        all_stds = [v['std'] for v in results['groups'].values()]
        
        results['summary'] = {
            'n_groups': len(results['groups']),
            'mean_of_means': np.mean(all_means),
            'std_of_means': np.std(all_means),
            'mean_of_stds': np.mean(all_stds),
            'cv_between_groups': np.std(all_means) / np.mean(all_means) if np.mean(all_means) != 0 else 0
        }
        
        return results
    
    def calculate_correlations(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix with significance tests"""
        n = len(columns)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if col1 in df.columns and col2 in df.columns:
                    # Get common non-null indices
                    mask = df[col1].notna() & df[col2].notna()
                    data1 = df.loc[mask, col1]
                    data2 = df.loc[mask, col2]
                    
                    if len(data1) > 2:
                        corr, p_val = stats.pearsonr(data1, data2)
                        corr_matrix[i, j] = corr
                        p_matrix[i, j] = p_val
                    else:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
        
        # Create DataFrame
        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        
        return corr_df
    
    def perform_normality_tests(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Perform normality tests on specified columns"""
        results = []
        
        for col in columns:
            if col in df.columns:
                data = df[col].dropna()
                
                if len(data) >= 8:  # Minimum for tests
                    # Shapiro-Wilk test
                    if len(data) <= 5000:  # Shapiro-Wilk limit
                        sw_stat, sw_p = stats.shapiro(data)
                    else:
                        sw_stat, sw_p = np.nan, np.nan
                    
                    # Anderson-Darling test
                    ad_result = stats.anderson(data)
                    ad_stat = ad_result.statistic
                    ad_critical = ad_result.critical_values[2]  # 5% significance
                    
                    # Jarque-Bera test
                    jb_stat, jb_p = stats.jarque_bera(data)
                    
                    results.append({
                        'column': col,
                        'n_samples': len(data),
                        'shapiro_stat': sw_stat,
                        'shapiro_p': sw_p,
                        'shapiro_normal': sw_p > 0.05 if not np.isnan(sw_p) else None,
                        'anderson_stat': ad_stat,
                        'anderson_critical': ad_critical,
                        'anderson_normal': ad_stat < ad_critical,
                        'jarque_bera_stat': jb_stat,
                        'jarque_bera_p': jb_p,
                        'jarque_bera_normal': jb_p > 0.05
                    })
        
        return pd.DataFrame(results)
    
    def calculate_effect_sizes(self, df: pd.DataFrame, group_col: str, 
                             value_col: str) -> Dict:
        """Calculate effect sizes between groups"""
        if group_col not in df.columns or value_col not in df.columns:
            return {}
        
        groups = df.groupby(group_col)[value_col].apply(lambda x: x.dropna().values)
        group_names = list(groups.index)
        
        effect_sizes = {}
        
        # Calculate pairwise effect sizes
        for i, group1_name in enumerate(group_names):
            for j, group2_name in enumerate(group_names):
                if i < j:
                    group1 = groups[group1_name]
                    group2 = groups[group2_name]
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # Cohen's d
                        mean_diff = np.mean(group1) - np.mean(group2)
                        pooled_std = np.sqrt(
                            ((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2)
                        )
                        
                        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                        
                        # Interpretation
                        if abs(cohens_d) < 0.2:
                            interpretation = 'negligible'
                        elif abs(cohens_d) < 0.5:
                            interpretation = 'small'
                        elif abs(cohens_d) < 0.8:
                            interpretation = 'medium'
                        else:
                            interpretation = 'large'
                        
                        effect_sizes[f'{group1_name}_vs_{group2_name}'] = {
                            'cohens_d': cohens_d,
                            'interpretation': interpretation,
                            'mean_diff': mean_diff,
                            'pooled_std': pooled_std
                        }
        
        return effect_sizes
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, column: str, 
                                   method: str = 'iqr') -> pd.Series:
        """Detect statistical anomalies using various methods"""
        if column not in df.columns:
            return pd.Series()
        
        data = df[column].copy()
        
        if method == 'iqr':
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(data.dropna()))
            anomalies = pd.Series(False, index=data.index)
            anomalies[data.notna()] = z_scores > 3
            
        elif method == 'isolation_forest':
            # Would require sklearn.ensemble.IsolationForest
            # Placeholder for now
            anomalies = pd.Series(False, index=data.index)
            
        else:
            anomalies = pd.Series(False, index=data.index)
        
        return anomalies