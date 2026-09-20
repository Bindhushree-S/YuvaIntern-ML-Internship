# Week 4 – Model Evaluation and Validation Techniques

This repository contains the practical implementation for Week 4 of the YuvaIntern Machine Learning Internship.

## Objective
Evaluate a machine learning model using appropriate performance metrics, cross-validation techniques, and error analysis.

## Project Structure
- `evaluate_model.py` – Train/test split, classification metrics, confusion matrix, and error analysis.
- `cross_validation.py` – Stratified K-Fold cross-validation.
- `error_analysis.py` – Detailed analysis of incorrect predictions.
- `requirements.txt` – Required Python packages.
- `.gitignore` – Files and folders excluded from Git.

## Dataset
The scripts use the Breast Cancer Wisconsin dataset available directly through `sklearn.datasets.load_breast_cancer()`. No external dataset download is required.

## Metrics Covered
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Validation
- Train-test split
- Stratified K-Fold cross-validation

## Run
```bash
python -m pip install -r requirements.txt
python evaluate_model.py
python cross_validation.py
python error_analysis.py
```

## Notes
The scripts print actual metrics at runtime. No fabricated performance values are included in the report or repository.
