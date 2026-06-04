#!/usr/bin/env python3
"""
Student Performance Prediction - Main Pipeline
This script runs the complete ML pipeline for predicting student grades.
"""

import os
import sys
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator, FeatureImportanceAnalyzer
from visualization import Visualizer
from prediction import StudentPerformancePredictor

def create_directories():
    """Create output directories if they don't exist."""
    directories = ['../results', '../results/confusion_matrices', '../results/performance_metrics', '../results/visualizations', '../data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Output directories created")

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 70)
    print("STUDENT PERFORMANCE PREDICTION - ML PIPELINE")
    print("=" * 70)
    
    create_directories()
    
    # STEP 1: DATA PREPROCESSING
    print("\n\n🔄 STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    data_path = '../data/student_data.csv'
    if not os.path.exists(data_path):
        print(f"✗ Data file not found at {data_path}")
        print("📝 Please add your CSV file to the 'data/' directory")
        print("Expected columns: nationality, grade_level, hands_raised, attendances, hours_studied, marks")
        return
    
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    feature_names = preprocessor.get_feature_names()
    scaler = preprocessor.scaler
    
    # STEP 2: MODEL TRAINING
    print("\n\n🤖 STEP 2: MODEL TRAINING")
    print("=" * 70)
    trainer = ModelTrainer()
    trainer.initialize_models()
    trained_models = trainer.train_models(X_train, y_train)
    
    # STEP 3: MODEL EVALUATION
    print("\n\n📊 STEP 3: MODEL EVALUATION")
    print("=" * 70)
    evaluator = ModelEvaluator()
    evaluator.evaluate_multiple_models(trained_models, X_test, y_test)
    results_df = evaluator.compare_models()
    
    # STEP 4: VISUALIZATION
    print("\n\n📈 STEP 4: VISUALIZATION")
    print("=" * 70)
    visualizer = Visualizer()
    print("\n⏳ Generating visualizations...")
    
    visualizer.plot_model_comparison(results_df, 'Accuracy', '../results/visualizations/model_accuracy_comparison.png')
    visualizer.plot_multiple_metrics(results_df, '../results/visualizations/all_metrics_comparison.png')
    
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred, model_name, f'../results/confusion_matrices/{model_name}_cm.png')
    
    visualizer.plot_correlation_heatmap(preprocessor.df, '../results/visualizations/correlation_heatmap.png')
    visualizer.plot_class_distribution(y_test, '../results/visualizations/class_distribution.png')
    print("✓ Visualizations saved to results/visualizations/")
    
    # STEP 5: FEATURE IMPORTANCE
    print("\n\n🎯 STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    feature_analyzer = FeatureImportanceAnalyzer()
    feature_rankings = feature_analyzer.analyze_features(trained_models, feature_names)
    
    for model_name, imp_df in feature_rankings.items():
        print(f"\n{'-' * 50}")
        print(f"Feature Importance - {model_name}")
        print(f"{'-' * 50}")
        print(imp_df.to_string(index=False))
        visualizer.plot_feature_importance(imp_df, model_name, f'../results/visualizations/{model_name}_importance.png')
    
    # STEP 6: SAMPLE PREDICTIONS
    print("\n\n🔮 STEP 6: SAMPLE PREDICTIONS")
    print("=" * 70)
    predictor = StudentPerformancePredictor(trained_models, scaler, feature_names)
    
    sample_student = {
        'nationality': preprocessor.df['nationality'].iloc[0],
        'grade_level': preprocessor.df['grade_level'].iloc[0],
        'hands_raised': preprocessor.df['hands_raised'].iloc[0],
        'attendances': preprocessor.df['attendances'].iloc[0],
        'hours_studied': preprocessor.df['hours_studied'].iloc[0]
    }
    
    print("\n📚 Sample Student Data:")
    for feature, value in sample_student.items():
        print(f"  • {feature}: {value}")
    
    print("\n🔮 Predictions:")
    ensemble_result = predictor.predict_ensemble(sample_student)
    print(f"  • Ensemble Prediction: {ensemble_result['ensemble_prediction']}")
    print(f"\n  Individual Model Predictions:")
    for model_name, pred in ensemble_result['individual_predictions'].items():
        print(f"    - {model_name}: {pred}")
    
    # FINAL SUMMARY
    print("\n\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📁 Output Files Generated:")
    print("  • Confusion Matrices: results/confusion_matrices/")
    print("  • Visualizations: results/visualizations/")
    print("  • Performance Metrics: results/performance_metrics/")
    print("\n📊 Best Performing Model:")
    best_model = results_df['Accuracy'].idxmax()
    best_accuracy = results_df['Accuracy'].max()
    print(f"  • {best_model} with Accuracy: {best_accuracy:.4f}")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
