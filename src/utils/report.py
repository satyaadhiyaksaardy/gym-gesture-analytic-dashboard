"""
Report generation module
"""

from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional
from config import REPORT_TIMESTAMP_FORMAT, REPORT_FILENAME_FORMAT, IRREGULAR_SAMPLING_THRESHOLD_MS


class ReportGenerator:
    """Class for generating analysis reports"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.irregular_threshold = IRREGULAR_SAMPLING_THRESHOLD_MS
    
    def generate_report(self, df: pd.DataFrame, outlier_mask: Optional[pd.Series],
                       ml_results: Dict, data_loader, stats_analyzer, timing_analyzer) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("# Gym Movement Sensor Data Analysis Report")
        report.append(f"Generated on: {self.timestamp.strftime(REPORT_TIMESTAMP_FORMAT)}")
        report.append("\n" + "="*60 + "\n")
        
        # Dataset Overview
        report.append("## Dataset Overview")
        data_info = data_loader.get_data_info(df)
        report.append(f"- Total Samples (Raw): {data_info['total_rows']:,}")
        
        if outlier_mask is not None:
            clean_samples = len(df) - outlier_mask.sum()
            report.append(f"- Total Samples (Cleaned): {clean_samples:,}")
            report.append(f"- Outliers Removed: {outlier_mask.sum():,} ({outlier_mask.sum()/len(df)*100:.2f}%)")
        
        report.append(f"- Number of Columns: {data_info['total_columns']}")
        report.append(f"- Memory Usage: {data_info['memory_usage_mb']:.2f} MB")
        
        if data_info.get('unique_athlete_id'):
            report.append(f"- Number of Athletes: {data_info['unique_athlete_id']}")
        if data_info.get('unique_exercise_type'):
            report.append(f"- Exercise Types: {data_info['unique_exercise_type']}")
        
        report.append(f"- Missing Data Percentage: {data_info['missing_data_pct']:.2f}%")
        report.append(f"- Has Timestamp Data: {'Yes' if data_info['has_timestamp'] else 'No'}")
        
        # Data Quality Assessment
        report.append("\n## Data Quality Assessment")
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(df, data_info, outlier_mask)
        report.append(f"- Overall Data Quality Score: {quality_score:.1f}/100")
        
        if data_info['missing_data_pct'] < 5:
            report.append("- ✅ Missing Data: Good (< 5%)")
        elif data_info['missing_data_pct'] < 20:
            report.append("- ⚠️  Missing Data: Moderate (5-20%)")
        else:
            report.append("- ❌ Missing Data: Poor (> 20%)")
        
        if outlier_mask is not None:
            outlier_pct = outlier_mask.sum() / len(df) * 100
            if outlier_pct < 1:
                report.append("- ✅ Outliers: Minimal (< 1%)")
            elif outlier_pct < 5:
                report.append("- ⚠️  Outliers: Moderate (1-5%)")
            else:
                report.append("- ❌ Outliers: High (> 5%)")
        
        # Sensor Statistics
        report.append("\n## Sensor Statistics")
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        if available_sensors:
            # Accelerometer range
            acc_cols = [col for col in ['ax', 'ay', 'az'] if col in df.columns]
            if acc_cols:
                acc_min = df[acc_cols].min().min()
                acc_max = df[acc_cols].max().max()
                report.append(f"- Accelerometer Range: [{acc_min:.2f}, {acc_max:.2f}] g")
            
            # Gyroscope range
            gyro_cols = [col for col in ['gx', 'gy', 'gz'] if col in df.columns]
            if gyro_cols:
                gyro_min = df[gyro_cols].min().min()
                gyro_max = df[gyro_cols].max().max()
                report.append(f"- Gyroscope Range: [{gyro_min:.2f}, {gyro_max:.2f}] deg/s")
        
        # Machine Learning Results
        if ml_results and ml_results.get('X_scaled') is not None:
            report.append("\n## Machine Learning Analysis")
            
            if ml_results.get('sil_score', 0) > 0:
                report.append(f"- K-Means Silhouette Score: {ml_results['sil_score']:.3f}")
                
                # Interpret silhouette score
                if ml_results['sil_score'] > 0.5:
                    report.append("  → Good cluster separation")
                elif ml_results['sil_score'] > 0.25:
                    report.append("  → Moderate cluster separation")
                else:
                    report.append("  → Poor cluster separation")
            
            if ml_results.get('kmeans_labels') is not None:
                n_clusters = len(np.unique(ml_results['kmeans_labels']))
                report.append(f"- Number of K-Means Clusters: {n_clusters}")
            
            if ml_results.get('dbscan_labels') is not None:
                noise_count = np.sum(ml_results['dbscan_labels'] == -1)
                total_points = len(ml_results['dbscan_labels'])
                noise_pct = noise_count / total_points * 100
                report.append(f"- DBSCAN Noise Points: {noise_count} ({noise_pct:.1f}%)")
            
            if ml_results.get('pca') is not None:
                exp_var = ml_results['pca'].explained_variance_ratio_
                report.append("\n### PCA Results")
                for i, var in enumerate(exp_var[:3]):  # First 3 components
                    report.append(f"- PC{i+1} Explained Variance: {var:.3f} ({var*100:.1f}%)")
                
                cumvar = np.cumsum(exp_var)
                n_components_95 = np.argmax(cumvar >= 0.95) + 1
                report.append(f"- Components for 95% variance: {n_components_95}")
        else:
            report.append("\n## Machine Learning Analysis")
            report.append("- Not performed")
        
        # Timing Analysis
        if 'timestamp' in df.columns:
            report.append("\n## Timing Analysis")
            
            # Calculate basic timing stats
            intervals = df.groupby(['athlete_id', 'exercise_type', 'rep_number'])['timestamp'].apply(
                lambda x: x.diff().mean() if len(x) > 1 else None
            ).dropna()
            
            if len(intervals) > 0:
                report.append(f"- Mean Sampling Interval: {intervals.mean():.2f} ms")
                report.append(f"- Std Sampling Interval: {intervals.std():.2f} ms")
                
                sampling_df = timing_analyzer.analyze_sampling_consistency(df)
                irregular_count = (sampling_df['std_interval_ms'] > self.irregular_threshold).sum()
                irregular_pct = irregular_count / len(sampling_df) * 100
                report.append(f"- Groups with Irregular Sampling: {irregular_count} ({irregular_pct:.1f}%)")
        
        # Summary and Recommendations
        report.append("\n## Summary and Recommendations")
        
        recommendations = self._generate_recommendations(df, data_info, outlier_mask, ml_results, timing_analyzer)
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # Analysis Status
        report.append("\n## Analysis Status")
        report.append("- ✅ Data Loading: Complete")
        report.append(f"- {'✅' if outlier_mask is not None else '❌'} Outlier Detection: {'Complete' if outlier_mask is not None else 'Not performed'}")
        report.append(f"- {'✅' if ml_results.get('features_df') is not None else '❌'} Feature Extraction: {'Complete' if ml_results.get('features_df') is not None else 'Not performed'}")
        report.append(f"- {'✅' if ml_results.get('X_scaled') is not None else '❌'} Clustering Analysis: {'Complete' if ml_results.get('X_scaled') is not None else 'Not performed'}")
        report.append(f"- {'✅' if ml_results.get('pca') is not None else '❌'} PCA Analysis: {'Complete' if ml_results.get('pca') is not None else 'Not performed'}")
        
        # Footer
        report.append("\n" + "="*60)
        report.append("\nThis report was generated automatically by the Gym Movement Sensor Analysis Dashboard.")
        report.append("For detailed visualizations and interactive analysis, please use the dashboard interface.")
        
        return "\n".join(report)
    
    def _calculate_quality_score(self, df: pd.DataFrame, data_info: Dict, 
                                outlier_mask: Optional[pd.Series]) -> float:
        """Calculate overall data quality score"""
        score = 100.0
        
        # Deduct for missing data
        missing_pct = data_info['missing_data_pct']
        if missing_pct > 0:
            score -= min(30, missing_pct * 1.5)  # Max 30 point deduction
        
        # Deduct for outliers
        if outlier_mask is not None:
            outlier_pct = outlier_mask.sum() / len(df) * 100
            score -= min(20, outlier_pct * 2)  # Max 20 point deduction
        
        # Deduct for missing timestamp
        if not data_info['has_timestamp']:
            score -= 10
        
        # Deduct for insufficient data
        if data_info['total_rows'] < 1000:
            score -= 10
        elif data_info['total_rows'] < 100:
            score -= 20
        
        return max(0, score)
    
    def _generate_recommendations(self, df: pd.DataFrame, data_info: Dict,
                                 outlier_mask: Optional[pd.Series], 
                                 ml_results: Dict, timing_analyzer) -> list:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        if data_info['missing_data_pct'] > 5:
            recommendations.append("Consider imputing or removing rows with missing data to improve analysis quality")
        
        if outlier_mask is not None and outlier_mask.sum() / len(df) > 0.05:
            recommendations.append("High outlier rate detected - review data collection process for sensor errors")
        
        # Timing recommendations
        if 'timestamp' in df.columns:
            intervals = df.groupby(['athlete_id', 'exercise_type', 'rep_number'])['timestamp'].apply(
                lambda x: x.diff().std() if len(x) > 1 else None
            ).dropna()
            
            sampling_df = timing_analyzer.analyze_sampling_consistency(df)
            irregular_count = (sampling_df['std_interval_ms'] > self.irregular_threshold).sum()
            irregular_pct = irregular_count / len(sampling_df) * 100
            
            if irregular_count > 0 and irregular_pct > 15:
                recommendations.append("Irregular sampling detected - check sensor connectivity and data transmission")
        
        # ML recommendations
        if ml_results.get('sil_score', 0) < 0.25 and ml_results.get('sil_score', 0) > 0:
            recommendations.append("Poor cluster separation - consider feature engineering or different clustering algorithms")
        
        if ml_results.get('dbscan_labels') is not None:
            noise_pct = np.sum(ml_results['dbscan_labels'] == -1) / len(ml_results['dbscan_labels']) * 100
            if noise_pct > 10:
                recommendations.append("High DBSCAN noise rate - investigate anomalous movement patterns or sensor issues")
        
        # General recommendations
        if data_info['total_rows'] < 1000:
            recommendations.append("Limited data available - collect more samples for robust analysis")
        
        if not data_info['has_timestamp']:
            recommendations.append("Add timestamp data to enable timing and sampling rate analysis")
        
        if len(recommendations) == 0:
            recommendations.append("Data quality is good - proceed with detailed analysis")
        
        return recommendations
    
    def get_report_filename(self) -> str:
        """Generate report filename with timestamp"""
        return self.timestamp.strftime(REPORT_FILENAME_FORMAT) + ".txt"
    
    def generate_anomaly_report(self, df: pd.DataFrame, ml_results: Dict,
                               noise_analysis: Dict, timing_analysis: Dict) -> str:
        """Generate specialized anomaly investigation report"""
        report = []
        report.append("# Anomaly Investigation Report")
        report.append(f"Generated on: {self.timestamp.strftime(REPORT_TIMESTAMP_FORMAT)}")
        report.append("\n" + "="*60 + "\n")
        
        # DBSCAN Noise Analysis
        if noise_analysis:
            report.append("## DBSCAN Noise Analysis")
            report.append(f"- Total noise points: {noise_analysis['noise_count']} ({noise_analysis['noise_percentage']:.1f}%)")
            
            if noise_analysis.get('noise_by_exercise'):
                report.append("\n### Noise by Exercise Type")
                for exercise, stats in noise_analysis['noise_by_exercise'].items():
                    noise_rate = stats['mean'] * 100
                    report.append(f"- {exercise}: {stats['sum']}/{stats['count']} ({noise_rate:.1f}%)")
            
            if noise_analysis.get('noise_by_athlete'):
                report.append("\n### Top Athletes with Noise")
                # Sort by noise rate
                sorted_athletes = sorted(noise_analysis['noise_by_athlete'].items(), 
                                       key=lambda x: x[1]['mean'], reverse=True)[:5]
                for athlete, stats in sorted_athletes:
                    noise_rate = stats['mean'] * 100
                    report.append(f"- Athlete {athlete}: {stats['sum']}/{stats['count']} ({noise_rate:.1f}%)")
        
        # Timing Anomalies
        if timing_analysis:
            report.append("\n## Timing Anomalies")
            if 'irregular_groups' in timing_analysis:
                report.append(f"- Irregular sampling groups: {len(timing_analysis['irregular_groups'])}")
                
                if 'worst_group' in timing_analysis:
                    worst = timing_analysis['worst_group']
                    report.append(f"\n### Worst Sampling Group")
                    report.append(f"- Athlete: {worst['athlete_id']}")
                    report.append(f"- Exercise: {worst['exercise_type']}")
                    report.append(f"- STD Interval: {worst['std_interval_ms']:.2f} ms")
        
        report.append("\n" + "="*60)
        return "\n".join(report)