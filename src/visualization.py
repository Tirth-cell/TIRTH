#!/usr/bin/env python3
"""
Visualization Module
Create plots and visualizations for data and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self):
        """Initialize visualizer with default style."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_model_comparison(self, results_df, metric, save_path):
        """Plot model comparison for a specific metric."""
        plt.figure(figsize=(12, 6))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
        bars = plt.bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black', linewidth=1.5)
        plt.title(f'Model {metric} Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel(metric, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_multiple_metrics(self, results_df, save_path):
        """Plot multiple metrics comparison."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
            axes[idx].bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Model', fontsize=10)
            axes[idx].set_ylabel(metric, fontsize=10)
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=sorted(np.unique(y_true)),
                   yticklabels=sorted(np.unique(y_true)))
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, df, save_path):
        """Plot correlation heatmap."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, cbar=True, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_class_distribution(self, y, save_path):
        """Plot class distribution."""
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(y, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        bars = plt.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Grade Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, df_importance, model_name, save_path):
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_importance)))
        bars = plt.barh(df_importance['Feature'], df_importance['Importance'], color=colors, edgecolor='black', linewidth=1.5)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
