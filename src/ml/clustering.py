"""
Machine Learning clustering module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import streamlit as st
from typing import Tuple, Optional, Dict
import math
from config import (
    CLUSTERING_MIN_SAMPLES, KMEANS_MAX_K, KMEANS_N_INIT,
    DBSCAN_EPS_PERCENTILE_DEFAULT, DBSCAN_MIN_SAMPLES_DEFAULT,
    PCA_N_COMPONENTS, METADATA_COLS
)


class ClusterAnalyzer:
    """Class for performing clustering analysis on sensor features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.dbscan = None
        self.pca = None
    
    @st.cache_data
    def perform_clustering(_self, features_df: pd.DataFrame, 
                         eps_percentile: int = DBSCAN_EPS_PERCENTILE_DEFAULT,
                         min_samples: Optional[int] = None,
                         k_max: int = KMEANS_MAX_K) -> Dict:
        """
        Perform K-Means and DBSCAN clustering
        
        Args:
            features_df: DataFrame of extracted features
            eps_percentile: Percentile for DBSCAN eps calculation
            min_samples: Minimum samples for DBSCAN
            k_max: Maximum k for K-Means search
            
        Returns:
            Dictionary with clustering results
        """
        results = {
            'X_scaled': None,
            'kmeans_labels': None,
            'dbscan_labels': None,
            'sil_score': 0,
            'best_k': 2,
            'eps': 0,
            'min_samples': min_samples
        }
        
        # Select numeric features only
        feature_cols = [c for c in features_df.columns if c not in METADATA_COLS]
        if not feature_cols:
            return results
        
        # Extract data and check sample size
        X = features_df[feature_cols].fillna(0)
        n_samples, n_features = X.shape
        
        if n_samples < CLUSTERING_MIN_SAMPLES:
            return results
        
        # Remove zero-variance features
        variances = X.var()
        non_zero_var_cols = variances[variances > 0].index
        X = X[non_zero_var_cols]
        
        if X.shape[1] == 0:
            return results
        
        # Standardize features
        X_scaled = _self.scaler.fit_transform(X)
        results['X_scaled'] = X_scaled
        
        # K-Means clustering with automatic k selection
        best_k, sil_score = _self._choose_optimal_k(X_scaled, k_max)
        results['best_k'] = best_k
        results['sil_score'] = sil_score
        
        _self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=KMEANS_N_INIT)
        results['kmeans_labels'] = _self.kmeans.fit_predict(X_scaled)
        
        # DBSCAN clustering with adaptive parameters
        eps = _self._calculate_adaptive_eps(X_scaled, eps_percentile)
        results['eps'] = eps
        
        if min_samples is None:
            min_samples = max(n_features + 1, int(math.log(n_samples)))
        results['min_samples'] = min_samples
        
        _self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        results['dbscan_labels'] = _self.dbscan.fit_predict(X_scaled)
        
        return results
    
    def _choose_optimal_k(self, X_scaled: np.ndarray, k_max: int) -> Tuple[int, float]:
        """Choose optimal k for K-Means using silhouette score"""
        n_samples = len(X_scaled)
        k_max = min(k_max, n_samples - 1)
        
        best_k, best_score = 2, -1
        
        for k in range(2, k_max + 1):
            if k >= n_samples:
                break
                
            km = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
            labels = km.fit_predict(X_scaled)
            
            # Check if we have valid clusters
            if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
                try:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_k, best_score = k, score
                except:
                    continue
        
        return best_k, max(best_score, 0)
    
    def _calculate_adaptive_eps(self, X_scaled: np.ndarray, eps_percentile: int) -> float:
        """Calculate adaptive eps for DBSCAN using k-distance graph"""
        n_samples = len(X_scaled)
        k_nn = max(2, min(30, int(math.sqrt(n_samples))))
        
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        
        # Sort k-distances
        k_dist = np.sort(distances[:, -1])
        
        # Calculate eps from percentile
        eps = np.percentile(k_dist, eps_percentile)
        
        return eps
    
    @st.cache_data
    def perform_pca(_self, X_scaled: np.ndarray, n_components: int = PCA_N_COMPONENTS) -> Tuple[PCA, np.ndarray]:
        """
        Perform PCA analysis
        
        Args:
            X_scaled: Standardized feature array
            n_components: Number of components
            
        Returns:
            Tuple of (PCA object, transformed data)
        """
        if X_scaled is None or len(X_scaled) == 0:
            return None, None
        
        # Adjust n_components if necessary
        n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
        
        if n_components < 1:
            return None, None
        
        _self.pca = PCA(n_components=n_components)
        X_pca = _self.pca.fit_transform(X_scaled)
        
        return _self.pca, X_pca
    
    def get_cluster_statistics(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Get statistics for each cluster"""
        if labels is None:
            return pd.DataFrame()
        
        # Add cluster labels to features
        features_with_clusters = features_df.copy()
        features_with_clusters['cluster'] = labels
        
        # Group by cluster and calculate statistics
        cluster_stats = []
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points for DBSCAN
                continue
                
            cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features_df) * 100
            }
            
            # Add metadata statistics
            if 'exercise_type' in cluster_data.columns:
                stats['dominant_exercise'] = cluster_data['exercise_type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            
            if 'athlete_id' in cluster_data.columns:
                stats['n_athletes'] = cluster_data['athlete_id'].nunique()
            
            if 'weight_kg' in cluster_data.columns:
                stats['mean_weight'] = cluster_data['weight_kg'].mean()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def get_silhouette_scores_by_k(self, X_scaled: np.ndarray, k_range: range) -> Dict[int, float]:
        """Calculate silhouette scores for different k values"""
        scores = {}
        n_samples = len(X_scaled)
        
        for k in k_range:
            if k >= n_samples:
                break
                
            km = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
            labels = km.fit_predict(X_scaled)
            
            if len(np.unique(labels)) > 1:
                try:
                    scores[k] = silhouette_score(X_scaled, labels)
                except:
                    scores[k] = 0
            else:
                scores[k] = 0
        
        return scores
    
    def analyze_dbscan_noise(self, features_df: pd.DataFrame, dbscan_labels: np.ndarray) -> Dict:
        """Analyze characteristics of DBSCAN noise points"""
        if dbscan_labels is None:
            return {}
        
        noise_mask = dbscan_labels == -1
        noise_count = noise_mask.sum()
        
        analysis = {
            'noise_count': noise_count,
            'noise_percentage': noise_count / len(dbscan_labels) * 100,
            'noise_by_exercise': {},
            'noise_by_athlete': {}
        }
        
        if noise_count > 0:
            # Add noise labels to features
            features_with_noise = features_df.copy()
            features_with_noise['is_noise'] = noise_mask
            
            # Analyze by exercise type
            if 'exercise_type' in features_with_noise.columns:
                noise_by_exercise = features_with_noise.groupby('exercise_type')['is_noise'].agg(['sum', 'count', 'mean'])
                analysis['noise_by_exercise'] = noise_by_exercise.to_dict('index')
            
            # Analyze by athlete
            if 'athlete_id' in features_with_noise.columns:
                noise_by_athlete = features_with_noise.groupby('athlete_id')['is_noise'].agg(['sum', 'count', 'mean'])
                analysis['noise_by_athlete'] = noise_by_athlete.to_dict('index')
        
        return analysis