#!/usr/bin/env python3
"""
Data Preprocessing Module
Handles loading, cleaning, and preparing data for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        """Initialize preprocessor with data path."""
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load CSV data."""
        print(f"📂 Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        print(f"\n📋 Dataset Info:")
        print(self.df.head())
        return self.df
    
    def handle_missing_values(self):
        """Handle any missing values."""
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} missing values. Dropping rows...")
            self.df = self.df.dropna()
        else:
            print(f"✓ No missing values found")
    
    def encode_categorical_features(self):
        """Encode categorical features using LabelEncoder."""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"\n🔄 Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    def separate_features_and_target(self):
        """Separate features and target variable."""
        self.feature_names = [col for col in self.df.columns if col != 'marks']
        X = self.df[self.feature_names]
        y = self.df['marks']
        print(f"\n✓ Features: {self.feature_names}")
        print(f"✓ Target: marks")
        return X, y
    
    def scale_features(self, X):
        """Scale features using StandardScaler."""
        print(f"\n📊 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Features scaled using StandardScaler")
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print(f"\n✂️  Splitting data: {100-int(test_size*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self):
        """Run complete preprocessing pipeline."""
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_features()
        X, y = self.separate_features_and_target()
        X_scaled = self.scale_features(X)
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Return feature names."""
        return self.feature_names
