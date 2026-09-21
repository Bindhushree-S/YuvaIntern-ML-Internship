# Week 5 - Model Optimization and Experimentation

This folder contains the Week 5 optimization experiment for the YuvaIntern Machine Learning Engineer internship.

## Project
AI-Based Breast Cancer Classification using the scikit-learn Breast Cancer Wisconsin dataset.

## Methods
- Logistic Regression with Grid Search
- Random Forest with Randomized Search
- 5-fold Stratified Cross-Validation
- ROC-AUC as the primary tuning metric
- Final evaluation using accuracy, precision, recall, F1-score and ROC-AUC

## Reproducibility
```bash
pip install -r requirements.txt
python src/optimize_models.py
```

The script uses `random_state=42` and writes the selected parameters and metrics to `results/optimization_results.json`.

## Data
The dataset is loaded directly from `sklearn.datasets.load_breast_cancer`, so no private or proprietary data is included.
