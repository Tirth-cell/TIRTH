#!/usr/bin/env python3
"""
Model Training Module
Train multiple ML classifiers.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np

class ModelTrainer:
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.trained_models = {}
    
    def initialize_models(self):
        """Initialize all ML models."""
        print("🤖 Initializing ML Models...\n")
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Support Vector Machine': SVC(kernel='rbf', random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
        }
        
        for model_name in self.models.keys():
            print(f"  ✓ {model_name}")
    
    def train_models(self, X_train, y_train):
        """Train all models."""
        print("\n📚 Training Models...\n")
        
        for model_name, model in self.models.items():
            print(f"  Training {model_name}...", end=" ")
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            print(f"✓")
        
        print(f"\n✓ All {len(self.trained_models)} models trained successfully!")
        return self.trained_models
