# Tumor Detection Machine Learning Project

## Overview
This project implements a machine learning model to classify breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin Dataset. It includes model training, visualization, and evaluation.

## Files
- `tumor_detection_models.py`: Trains Logistic Regression, SVM, and Random Forest models.
- `tumor_detection_visualize.py`: Generates a decision boundary plot.
- `tumor_detection_final.py`: Evaluates the best model (Random Forest).
- `best_model.pkl`: Saved Random Forest model.
- `scaler.pkl`: Saved scaler for data consistency.
- `results.txt`: Final evaluation metrics.
- `decision_boundary.png`: Visualization of the decision boundary.
- `Tumor_Detection_Report.pdf`: Detailed project report.

## Results
- Best Model Accuracy: 0.96
- Classification Report: Precision 0.98 (Class 0), 0.96 (Class 1); Recall 0.93 (Class 0), 0.99 (Class 1).

## How to Run
1. Install dependencies: `pip install scikit-learn pandas numpy matplotlib joblib`.
2. Run `tumor_detection_models.py` to train models.
3. Run `tumor_detection_visualize.py` for visualization.
4. Run `tumor_detection_final.py` for final evaluation.


