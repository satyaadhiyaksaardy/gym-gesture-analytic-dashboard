"""
Machine Learning visualization module
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Dict, List
from config import PLOT_STYLE, FIGURE_DPI, COLOR_PALETTE
from utils.helpers import safe_plot_close


class MLVisualizer:
    """Class for creating machine learning visualizations"""
    
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.dpi = FIGURE_DPI
        self.palette = COLOR_PALETTE
    
    def plot_silhouette_analysis(self, silhouette_scores: Dict[int, float]) -> plt.Figure:
        """Plot silhouette scores for different k values"""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        k_values = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())
        
        ax.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score vs Number of Clusters')
        ax.grid(True, alpha=0.3)
        
        # Mark the best k
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]
        ax.plot(best_k, best_score, 'ro', markersize=12)
        ax.annotate(f'Best k={best_k}\nScore={best_score:.3f}',
                   xy=(best_k, best_score), xytext=(best_k+0.5, best_score-0.05),
                   fontsize=10, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_distribution(self, kmeans_labels: np.ndarray, 
                                 dbscan_labels: np.ndarray) -> plt.Figure:
        """Plot cluster distribution for K-Means and DBSCAN"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # K-Means distribution
        if kmeans_labels is not None:
            unique_kmeans, counts_kmeans = np.unique(kmeans_labels, return_counts=True)
            ax1.bar(unique_kmeans, counts_kmeans, color='steelblue', alpha=0.7)
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Number of Points')
            ax1.set_title('K-Means Cluster Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (cluster, count) in enumerate(zip(unique_kmeans, counts_kmeans)):
                ax1.text(cluster, count + max(counts_kmeans)*0.01, str(count),
                        ha='center', va='bottom')
        
        # DBSCAN distribution
        if dbscan_labels is not None:
            unique_dbscan, counts_dbscan = np.unique(dbscan_labels, return_counts=True)
            colors = ['red' if label == -1 else 'green' for label in unique_dbscan]
            ax2.bar(unique_dbscan, counts_dbscan, color=colors, alpha=0.7)
            ax2.set_xlabel('Cluster ID (-1 = Noise)')
            ax2.set_ylabel('Number of Points')
            ax2.set_title('DBSCAN Cluster Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (cluster, count) in enumerate(zip(unique_dbscan, counts_dbscan)):
                ax2.text(cluster, count + max(counts_dbscan)*0.01, str(count),
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_pca_2d(self, X_pca: np.ndarray, labels: np.ndarray, 
                    explained_variance: np.ndarray,
                    title: str = "PCA 2D Visualization") -> plt.Figure:
        """Create 2D PCA scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        if X_pca.shape[1] >= 2:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=labels, cmap='viridis', 
                               alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
            
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster')
        
        plt.tight_layout()
        return fig
    
    def plot_pca_3d(self, X_pca: np.ndarray, labels: np.ndarray,
                    explained_variance: np.ndarray,
                    title: str = "PCA 3D Visualization") -> plt.Figure:
        """Create 3D PCA scatter plot"""
        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        if X_pca.shape[1] >= 3:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                               c=labels, cmap='viridis',
                               alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
            ax.set_title(title)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
            cbar.set_label('Cluster')
        
        plt.tight_layout()
        return fig
    
    def plot_explained_variance(self, explained_variance_ratio: np.ndarray) -> plt.Figure:
        """Plot PCA explained variance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        n_components = len(explained_variance_ratio)
        components = range(1, n_components + 1)
        
        # Individual explained variance
        ax1.bar(components, explained_variance_ratio, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, var in enumerate(explained_variance_ratio):
            ax1.text(i+1, var + max(explained_variance_ratio)*0.01, 
                    f'{var:.1%}', ha='center', va='bottom')
        
        # Cumulative explained variance
        cumvar = np.cumsum(explained_variance_ratio)
        ax2.plot(components, cumvar, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_comparison(self, features_df: pd.DataFrame, 
                               kmeans_labels: np.ndarray,
                               feature_names: List[str]) -> plt.Figure:
        """Compare feature distributions across clusters"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
        axes = axes.ravel()
        
        # Add cluster labels to features
        plot_df = features_df.copy()
        plot_df['cluster'] = kmeans_labels
        
        # Select top features to plot
        features_to_plot = feature_names[:4]
        
        for idx, feature in enumerate(features_to_plot):
            if idx < len(axes) and feature in plot_df.columns:
                ax = axes[idx]
                
                # Create box plot
                plot_df.boxplot(column=feature, by='cluster', ax=ax)
                ax.set_title(f'{feature} by Cluster')
                ax.set_xlabel('Cluster')
                ax.set_ylabel(feature)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distribution by Cluster', fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_dbscan_noise_analysis(self, features_df: pd.DataFrame,
                                   dbscan_labels: np.ndarray) -> plt.Figure:
        """Analyze DBSCAN noise points"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # Noise distribution by exercise type
        if 'exercise_type' in features_df.columns:
            noise_df = features_df.copy()
            noise_df['is_noise'] = dbscan_labels == -1
            
            noise_by_exercise = noise_df.groupby('exercise_type')['is_noise'].agg(['sum', 'count'])
            noise_by_exercise['percentage'] = noise_by_exercise['sum'] / noise_by_exercise['count'] * 100
            
            ax1.bar(range(len(noise_by_exercise)), noise_by_exercise['percentage'], 
                   color='coral', alpha=0.7)
            ax1.set_xticks(range(len(noise_by_exercise)))
            ax1.set_xticklabels(noise_by_exercise.index, rotation=45, ha='right')
            ax1.set_ylabel('Noise Percentage (%)')
            ax1.set_title('DBSCAN Noise Rate by Exercise Type')
            ax1.grid(True, alpha=0.3)
        
        # Noise distribution by athlete
        if 'athlete_id' in features_df.columns:
            noise_by_athlete = noise_df.groupby('athlete_id')['is_noise'].agg(['sum', 'count'])
            noise_by_athlete['percentage'] = noise_by_athlete['sum'] / noise_by_athlete['count'] * 100
            
            # Plot top 10 athletes with highest noise rate
            top_noise_athletes = noise_by_athlete.nlargest(10, 'percentage')
            
            ax2.bar(range(len(top_noise_athletes)), top_noise_athletes['percentage'],
                   color='skyblue', alpha=0.7)
            ax2.set_xticks(range(len(top_noise_athletes)))
            ax2.set_xticklabels(top_noise_athletes.index, rotation=45, ha='right')
            ax2.set_ylabel('Noise Percentage (%)')
            ax2.set_title('Top 10 Athletes by DBSCAN Noise Rate')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_heatmap(self, cluster_stats: pd.DataFrame) -> plt.Figure:
        """Create heatmap of cluster characteristics"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Select numeric columns for heatmap
        numeric_cols = cluster_stats.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Normalize data for better visualization
            heatmap_data = cluster_stats[numeric_cols].T
            heatmap_data_norm = (heatmap_data - heatmap_data.mean(axis=1).values.reshape(-1, 1)) / heatmap_data.std(axis=1).values.reshape(-1, 1)
            
            sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, cbar_kws={'label': 'Normalized Value'},
                       xticklabels=[f'Cluster {i}' for i in cluster_stats['cluster_id']],
                       yticklabels=numeric_cols, ax=ax)
            
            ax.set_title('Cluster Characteristics Heatmap (Normalized)')
        
        plt.tight_layout()
        return fig