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
import warnings
warnings.filterwarnings('ignore')


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
            try:
                adf_result = adfuller(clean_data, autolag='AIC')
                results['adf'] = {
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05,
                    'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
                }
            except Exception as e:
                results['adf'] = {'error': str(e)}
        
        # KPSS test
        if test_type in ['kpss', 'both']:
            try:
                # KPSS test with trend='c' (constant) - testing for level stationarity
                kpss_result = kpss(clean_data, regression='c', nlags="auto")
                results['kpss'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05,
                    'interpretation': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
                }
            except Exception as e:
                results['kpss'] = {'error': str(e)}
        
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
            try:
                sw_stat, sw_p = stats.shapiro(clean_data)
                results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_p),
                    'is_normal': sw_p > 0.05
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        # D'Agostino-Pearson test
        if len(clean_data) >= 20:
            try:
                da_stat, da_p = stats.normaltest(clean_data)
                results['dagostino_pearson'] = {
                    'statistic': float(da_stat),
                    'p_value': float(da_p),
                    'is_normal': da_p > 0.05
                }
            except Exception as e:
                results['dagostino_pearson'] = {'error': str(e)}
        
        # Anderson-Darling test
        try:
            ad_result = stats.anderson(clean_data, dist='norm')
            results['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], 
                                           [float(cv) for cv in ad_result.critical_values])),
                'is_normal_5pct': ad_result.statistic < ad_result.critical_values[2]
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # Q-Q plot correlation
        try:
            sorted_data = np.sort(clean_data)
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
            qq_corr = np.corrcoef(sorted_data, theoretical_quantiles)[0, 1]
            results['qq_correlation'] = float(qq_corr)
        except Exception as e:
            results['qq_correlation'] = 0.0
        
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
                group_names.append(str(name))
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for comparison'}
        
        # One-way ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'groups': group_names
            }
        except Exception as e:
            results['anova'] = {'error': str(e)}
        
        # Kruskal-Wallis (non-parametric alternative)
        try:
            h_stat, p_value_kw = stats.kruskal(*groups)
            results['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(p_value_kw),
                'significant': p_value_kw < 0.05
            }
        except Exception as e:
            results['kruskal_wallis'] = {'error': str(e)}
        
        # Post-hoc analysis if significant
        if results.get('anova', {}).get('significant', False):
            try:
                # Prepare data for Tukey HSD
                all_data = []
                all_groups = []
                for data, name in zip(groups, group_names):
                    all_data.extend(data)
                    all_groups.extend([name] * len(data))
                
                # Perform Tukey HSD
                tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
                
                # Extract results properly
                results['tukey_hsd'] = {
                    'summary': str(tukey_result),
                    'significant_pairs': []
                }
                
                # Parse the results - Fixed approach
                result_frame = tukey_result.summary().data
                if len(result_frame) > 1:  # Has data beyond header
                    headers = result_frame[0]
                    for row in result_frame[1:]:
                        # Find column indices
                        group1_idx = headers.index('group1')
                        group2_idx = headers.index('group2')
                        meandiff_idx = headers.index('meandiff')
                        pval_idx = headers.index('p-adj')
                        reject_idx = headers.index('reject')
                        
                        if row[reject_idx]:  # If reject is True
                            results['tukey_hsd']['significant_pairs'].append({
                                'group1': str(row[group1_idx]),
                                'group2': str(row[group2_idx]),
                                'mean_diff': float(row[meandiff_idx]),
                                'p_value': float(row[pval_idx])
                            })
            except Exception as e:
                results['tukey_hsd'] = {'error': str(e)}
        
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
                    try:
                        # Pearson correlation test
                        corr, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                        
                        # Spearman correlation (non-parametric)
                        spearman_corr, spearman_p = stats.spearmanr(
                            data1[common_idx], data2[common_idx]
                        )
                        
                        # Simple mutual information approximation using binning
                        # (avoiding sklearn dependency)
                        bins = 10
                        hist_2d, _, _ = np.histogram2d(
                            data1[common_idx], data2[common_idx], bins=bins
                        )
                        hist_2d = hist_2d / hist_2d.sum()
                        
                        # Marginal distributions
                        p_x = hist_2d.sum(axis=1)
                        p_y = hist_2d.sum(axis=0)
                        
                        # Mutual information calculation
                        mi = 0
                        for i in range(bins):
                            for j in range(bins):
                                if hist_2d[i, j] > 0:
                                    mi += hist_2d[i, j] * np.log2(
                                        hist_2d[i, j] / (p_x[i] * p_y[j] + 1e-10)
                                    )
                        
                        results[f'{sensor1}_vs_{sensor2}'] = {
                            'pearson_correlation': float(corr),
                            'pearson_p_value': float(p_value),
                            'spearman_correlation': float(spearman_corr),
                            'spearman_p_value': float(spearman_p),
                            'mutual_information': float(mi),
                            'linear_dependent': p_value < 0.05,
                            'monotonic_dependent': spearman_p < 0.05,
                            'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                        }
                    except Exception as e:
                        results[f'{sensor1}_vs_{sensor2}'] = {'error': str(e)}
        
        return results
    
    @st.cache_data
    def test_variance_homogeneity(_self, df: pd.DataFrame, 
                                 sensor_col: str,
                                 group_col: str = 'exercise_type') -> Dict:
        """
        Test homogeneity of variances across groups (Levene's test)
        
        Returns:
            Dictionary with variance homogeneity test results
        """
        if sensor_col not in df.columns or group_col not in df.columns:
            return {'error': 'Required columns not found'}
        
        # Prepare groups
        groups = []
        group_names = []
        for name, group in df.groupby(group_col):
            data = group[sensor_col].dropna()
            if len(data) > 5:
                groups.append(data.values)
                group_names.append(str(name))
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for comparison'}
        
        try:
            # Levene's test
            levene_stat, levene_p = stats.levene(*groups)
            
            # Bartlett's test (assumes normality)
            bartlett_stat, bartlett_p = stats.bartlett(*groups)
            
            # Calculate variance ratios
            variances = [np.var(g, ddof=1) for g in groups]
            max_var = max(variances)
            min_var = min(variances)
            var_ratio = max_var / min_var if min_var > 0 else np.inf
            
            return {
                'levene': {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'equal_variances': levene_p > 0.05
                },
                'bartlett': {
                    'statistic': float(bartlett_stat),
                    'p_value': float(bartlett_p),
                    'equal_variances': bartlett_p > 0.05
                },
                'variance_ratio': float(var_ratio),
                'group_variances': dict(zip(group_names, [float(v) for v in variances]))
            }
        except Exception as e:
            return {'error': str(e)}