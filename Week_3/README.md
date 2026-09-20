# YuvaIntern – Week 3: Model Implementation and Code Documentation

## AI-Based Customer Churn Prediction System

This repository contains the Week 3 implementation for the **YuvaIntern Machine Learning Engineer Internship**.

The project develops a supervised machine-learning system that predicts whether a telecommunications customer is likely to churn based on historical customer, account, service and billing information.

> **Important:** Model performance values are not hard-coded or fabricated. Run the training script to generate the actual validation and test metrics.

---

## 1. Problem Statement

Customer churn occurs when an existing customer stops using a service. The objective of this project is to learn patterns from historical customer records and predict whether a new customer is likely to churn.

The machine-learning problem is formulated as:

- **Learning type:** Supervised Learning
- **Problem type:** Binary Classification
- **Target:** Churn
- **Classes:** No Churn / Churn
- **Primary output:** Predicted class and churn probability

---

## 2. Objectives

The Week 3 implementation aims to:

1. Implement a complete machine-learning training workflow.
2. Compare an interpretable baseline with a non-linear ensemble model.
3. Build leakage-safe preprocessing.
4. Evaluate models using appropriate classification metrics.
5. Save the complete preprocessing + model pipeline.
6. Provide reusable prediction functionality.
7. Add automated tests for important functionality.
8. Follow maintainable coding, documentation and version-control practices.

---

## 3. Dataset

The project is designed for the public **IBM Telco Customer Churn** sample dataset.

Expected CSV filename:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place the file at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The raw dataset is intentionally not included in this repository structure. Refer to `data/README.md` for dataset placement guidance.

---

## 4. Models

### Logistic Regression

Logistic Regression is used as the baseline because it is:

- Simple
- Fast
- Interpretable
- Appropriate for binary classification

### Random Forest

Random Forest is used as the main non-linear candidate because it can:

- Capture non-linear relationships
- Capture feature interactions
- Handle mixed tabular features after preprocessing
- Provide useful feature-importance information
- Reduce the variance associated with a single decision tree

The final model should be selected based on validation evidence rather than an assumed accuracy.

---

## 5. Machine Learning Workflow

```text
Public Dataset
      |
      v
Data Loading
      |
      v
Data Validation
      |
      v
Data Cleaning
      |
      v
Train/Test Split
      |
      v
Preprocessing
      |
      +----------------------+
      |                      |
      v                      v
Logistic Regression     Random Forest
      |                      |
      +----------+-----------+
                 |
                 v
       Stratified Cross-Validation
                 |
                 v
         Model Comparison
                 |
                 v
        Held-Out Test Set
                 |
                 v
       Error Analysis + Metrics
                 |
                 v
      Save Selected Pipeline
```

---

## 6. Preprocessing

The implementation uses `ColumnTransformer` and `Pipeline`.

### Numerical features

- Missing values are imputed using the median.
- Standard scaling is applied.

### Categorical features

- Missing values are imputed using the most frequent value.
- One-hot encoding is applied.
- `handle_unknown="ignore"` prevents inference failures when an unseen category appears.

Using a pipeline ensures that preprocessing is learned only from training data and reused consistently during inference.

---

## 7. Evaluation Metrics

The implementation calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

For churn prediction, recall and F1-score are particularly important because a system that misses many actual churners may have limited business value.

---

## 8. Project Structure

```text
YuvaIntern-Week3/
|
+-- data/
|   +-- README.md
|   +-- raw/
|       +-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|
+-- src/
|   +-- train.py
|
+-- tests/
|   +-- test_pipeline.py
|
+-- models/
|   +-- churn_pipeline.joblib       # generated after training
|
+-- reports/
|   +-- Week_3_Model_Implementation_Report.docx
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## 9. Installation

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 10. Run the Training Program

After placing the dataset in the expected location:

```bash
python src/train.py
```

The script will:

1. Load the dataset.
2. Validate the target and basic data quality.
3. Convert `TotalCharges` safely to numeric.
4. Remove `customerID` from model features.
5. Create a stratified train/test split.
6. Build preprocessing pipelines.
7. Train and compare Logistic Regression and Random Forest.
8. Perform five-fold stratified ROC-AUC cross-validation.
9. Evaluate the selected model on the held-out test set.
10. Save the selected complete pipeline to:

```text
models/churn_pipeline.joblib
```

---

## 11. Run Tests

Run:

```bash
pytest -q
```

The tests verify important behaviours such as:

- Required target validation
- Binary target conversion
- Removal of identifier columns
- Model fitting
- Prediction output
- Probability range

---

## 12. Reproducibility

The project uses a fixed random seed:

```python
RANDOM_STATE = 42
```

This helps make the train/test split and stochastic model behaviour reproducible.

For stronger reproducibility, record:

- Python version
- Package versions
- Dataset version
- Git commit
- Model hyperparameters
- Cross-validation settings
- Evaluation metrics

---

## 13. Version Control

Recommended commit sequence:

```bash
git add .
git commit -m "Initialize Week 3 ML project"

git add src/train.py
git commit -m "Implement churn model training pipeline"

git add tests/test_pipeline.py
git commit -m "Add ML pipeline tests"

git add README.md requirements.txt .gitignore
git commit -m "Document Week 3 implementation"

git tag v0.1-week3
```

---

## 14. Error Handling

The implementation explicitly checks for:

- Missing dataset
- Empty dataset
- Missing `Churn` column
- Unexpected target labels
- Invalid target conversion
- Missing prediction fields

The goal is to fail early with a meaningful error rather than silently producing unreliable predictions.

---

## 15. Future Enhancements

Possible next steps include:

- Hyperparameter tuning
- Probability calibration
- SHAP-based explanations
- Feature importance visualization
- Threshold optimization
- FastAPI prediction endpoint
- Streamlit or React dashboard
- Model monitoring
- Data drift detection
- CI/CD testing using GitHub Actions

---

## 16. Academic Note

This repository represents an implementation-oriented ML workflow. The purpose of Week 3 is not only to train a classifier, but to demonstrate:

- Algorithm selection
- Practical implementation
- Data preprocessing
- Leakage prevention
- Validation
- Testing
- Debugging
- Reproducibility
- Maintainable code
- Technical documentation

Actual model metrics should always be generated by executing the code and should not be manually invented.
