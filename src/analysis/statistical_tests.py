"""
Statistical testing module for gesture analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
from typing import Dict, Tuple, List


class StatisticalTester:
    """Class for performing statistical tests on sensor data"""
    
    @st.cache_data
    def test_stationarity(_self, data: pd.Series, test_type: str = 'both') -> Dict:
        """
        Test signal stationarity using ADF and KPSS tests
        
        Args:
            data: Time series data
            test_type: 'adf', 'kpss', or 'both'
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 20:
            return {'error': 'Insufficient data for stationarity test'}
        
        # Augmented Dickey-Fuller test
        if test_type in ['adf', 'both']:
            adf_result = adfuller(clean_data, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05,
                'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
            }
        
        # KPSS test
        if test_type in ['kpss', 'both']:
            kpss_result = kpss(clean_data, regression='c', nlags="auto")
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05,
                'interpretation': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
            }
        
        return results
    
    @st.cache_data
    def test_normality(_self, data: pd.Series) -> Dict:
        """
        Comprehensive normality testing
        
        Returns:
            Dictionary with multiple normality test results
        """
        clean_data = data.dropna()
        
        if len(clean_data) < 8:
            return {'error': 'Insufficient data for normality tests'}
        
        results = {}
        
        # Shapiro-Wilk test
        if len(clean_data) <= 5000:
            sw_stat, sw_p = stats.shapiro(clean_data)
            results['shapiro_wilk'] = {
                'statistic': sw_stat,
                'p_value': sw_p,
                'is_normal': sw_p > 0.05
            }
        
        # D'Agostino-Pearson test
        if len(clean_data) >= 20:
            da_stat, da_p = stats.normaltest(clean_data)
            results['dagostino_pearson'] = {
                'statistic': da_stat,
                'p_value': da_p,
                'is_normal': da_p > 0.05
            }
        
        # Anderson-Darling test
        ad_result = stats.anderson(clean_data, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], 
                                       ad_result.critical_values)),
            'is_normal_5pct': ad_result.statistic < ad_result.critical_values[2]
        }
        
        # Q-Q plot correlation
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_data)))
        qq_corr = np.corrcoef(np.sort(clean_data), theoretical_quantiles)[0, 1]
        results['qq_correlation'] = qq_corr
        
        return results
    
    @st.cache_data
    def compare_exercise_patterns(_self, df: pd.DataFrame, 
                                sensor_col: str,
                                group_col: str = 'exercise_type') -> Dict:
        """
        Compare sensor patterns across different exercises
        
        Returns:
            Dictionary with ANOVA and post-hoc test results
        """
        if sensor_col not in df.columns or group_col not in df.columns:
            return {'error': 'Required columns not found'}
        
        results = {}
        
        # Prepare groups
        groups = []
        group_names = []
        for name, group in df.groupby(group_col):
            data = group[sensor_col].dropna()
            if len(data) > 5:
                groups.append(data.values)
                group_names.append(name)
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for comparison'}
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'groups': group_names
        }
        
        # Kruskal-Wallis (non-parametric alternative)
        h_stat, p_value_kw = stats.kruskal(*groups)
        results['kruskal_wallis'] = {
            'h_statistic': h_stat,
            'p_value': p_value_kw,
            'significant': p_value_kw < 0.05
        }
        
        # Post-hoc analysis if significant
        if results['anova']['significant']:
            # Prepare data for Tukey HSD
            all_data = []
            all_groups = []
            for i, (data, name) in enumerate(zip(groups, group_names)):
                all_data.extend(data)
                all_groups.extend([name] * len(data))
            
            # Tukey HSD
            tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
            
            results['tukey_hsd'] = {
                'summary': str(tukey_result),
                'significant_pairs': []
            }
            
            # Extract significant pairs
            for i in range(len(tukey_result.summary())):
                if i > 0:  # Skip header
                    row = tukey_result.summary()[i]
                    if row[5]:  # Reject null hypothesis
                        results['tukey_hsd']['significant_pairs'].append({
                            'group1': row[0],
                            'group2': row[1],
                            'mean_diff': float(row[2]),
                            'p_value': float(row[4])
                        })
        
        return results
    
    @st.cache_data
    def test_sensor_independence(_self, df: pd.DataFrame) -> Dict:
        """
        Test independence between sensor channels
        
        Returns:
            Dictionary with independence test results
        """
        sensor_pairs = [
            ('ax', 'ay'), ('ax', 'az'), ('ay', 'az'),
            ('gx', 'gy'), ('gx', 'gz'), ('gy', 'gz'),
            ('ax', 'gx'), ('ay', 'gy'), ('az', 'gz')
        ]
        
        results = {}
        
        for sensor1, sensor2 in sensor_pairs:
            if sensor1 in df.columns and sensor2 in df.columns:
                data1 = df[sensor1].dropna()
                data2 = df[sensor2].dropna()
                
                # Get common indices
                common_idx = data1.index.intersection(data2.index)
                
                if len(common_idx) > 30:
                    # Pearson correlation test
                    corr, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                    
                    # Mutual information (non-linear dependence)
                    from sklearn.feature_selection import mutual_info_regression
                    mi = mutual_info_regression(
                        data1[common_idx].values.reshape(-1, 1),
                        data2[common_idx].values,
                        random_state=42
                    )[0]
                    
                    results[f'{sensor1}_vs_{sensor2}'] = {
                        'pearson_correlation': corr,
                        'pearson_p_value': p_value,
                        'mutual_information': mi,
                        'linear_dependent': p_value < 0.05,
                        'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                    }
        
        return results