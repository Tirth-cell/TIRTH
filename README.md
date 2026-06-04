# Student Performance Prediction - Machine Learning Project

## Overview
This is a comprehensive machine learning project that uses various classifiers and AI models to predict factors affecting student grades. The project analyzes student data including demographics, behavioral factors, and academic metrics to identify key predictors of academic performance.

## Dataset Description
The project uses a CSV dataset containing student information with the following attributes:
- **Demographics**: Student nationality, grade level
- **Behavioral Factors**: Number of hands raised, class participation
- **Academic Metrics**: Number of absences/attendances, hours studied
- **Target Variable**: Student grades/marks

## Project Features
✅ Multiple ML classifiers for accurate predictions
✅ Data preprocessing and feature engineering
✅ Comprehensive data visualization (graphs, charts, heatmaps)
✅ Confusion matrices for model evaluation
✅ Correlation analysis and statistical insights
✅ Model comparison and performance metrics
✅ Feature importance analysis

## Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **XGBoost** - Advanced gradient boosting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tirth-cell/TIRTH.git
cd TIRTH
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the ML Pipeline
```bash
cd src
python main.py
```

### View Results
Check the `results/` directory for visualizations and metrics.

## ML Models Implemented
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **XGBoost Classifier**

## Project Structure
```
TIRTH/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── student_data.csv
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    ├── model_evaluation.py
    ├── visualization.py
    ├── prediction.py
    └── main.py
```

## Author
Tirth - Student Performance Prediction ML Project
