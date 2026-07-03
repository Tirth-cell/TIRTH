#!/usr/bin/env python3
"""
Prediction Module
Make predictions on new student data.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

class StudentPerformancePredictor:
    def __init__(self, trained_models, scaler, feature_names):
        """Initialize predictor with trained models and scaler."""
        self.trained_models = trained_models
        self.scaler = scaler
        self.feature_names = feature_names
        self.label_encoders = {
            'nationality': LabelEncoder(),
            'grade_level': LabelEncoder()
        }
        
        # Fit encoders with common values
        self.label_encoders['nationality'].fit(['Canada', 'Egypt', 'USA'])
        self.label_encoders['grade_level'].fit(['L', 'M', 'U'])
    
    def prepare_input(self, student_data):
        """Prepare input data for prediction."""
        # Create feature vector in correct order
        features = []
        for feature_name in self.feature_names:
            if feature_name == 'nationality':
                encoded = self.label_encoders['nationality'].transform([student_data['nationality']])[0]
                features.append(encoded)
            elif feature_name == 'grade_level':
                encoded = self.label_encoders['grade_level'].transform([student_data['grade_level']])[0]
                features.append(encoded)
            else:
                features.append(student_data[feature_name])
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict_ensemble(self, student_data):
        """Make prediction using ensemble of models."""
        X_scaled = self.prepare_input(student_data)
        
        predictions = {}
        for model_name, model in self.trained_models.items():
            pred = model.predict(X_scaled)[0]
            predictions[model_name] = pred
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        return {
            'ensemble_prediction': round(ensemble_pred, 2),
            'individual_predictions': predictions
        }
