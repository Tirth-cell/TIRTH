#!/usr/bin/env python3
"""
Model Evaluation Module
Evaluate and compare model performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class ModelEvaluator:
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
        self.predictions = {}
        self.confusion_matrices = {}
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        self.predictions[model_name] = y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        self.confusion_matrices[model_name] = cm
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    def evaluate_multiple_models(self, trained_models, X_test, y_test):
        """Evaluate all trained models."""
        print("📊 Evaluating Models...\n")
        
        for model_name, model in trained_models.items():
            result = self.evaluate_model(model_name, model, X_test, y_test)
            self.results.append(result)
            print(f"\n{'-' * 60}")
            print(f"Model: {model_name}")
            print(f"{'-' * 60}")
            print(f"  Accuracy:  {result['Accuracy']:.4f}")
            print(f"  Precision: {result['Precision']:.4f}")
            print(f"  Recall:    {result['Recall']:.4f}")
            print(f"  F1-Score:  {result['F1-Score']:.4f}")
    
    def compare_models(self):
        """Compare all models and return results dataframe."""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print(f"\n\n{'=' * 70}")
        print("MODEL COMPARISON")
        print(f"{'=' * 70}")
        print(results_df.to_string(index=False))
        
        return results_df

class FeatureImportanceAnalyzer:
    def __init__(self):
        """Initialize feature importance analyzer."""
        pass
    
    def analyze_features(self, trained_models, feature_names):
        """Analyze feature importance for tree-based models."""
        feature_rankings = {}
        tree_models = ['Decision Tree', 'Random Forest', 'XGBoost']
        
        for model_name, model in trained_models.items():
            if model_name in tree_models and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                df_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                feature_rankings[model_name] = df_importance
        
        return feature_rankings
