"""
Week 5: Model Optimization and Experimentation
Breast Cancer Wisconsin classification project.
"""
import json
import platform
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import randint

RANDOM_STATE = 42

def metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    }

def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])
    grid = GridSearchCV(
        lr,
        {"model__C":[0.01,0.1,1,10,100],
         "model__solver":["liblinear","lbfgs"]},
        scoring="roc_auc", cv=cv, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": randint(100,501),
         "max_depth":[None,5,10,15,20,25],
         "min_samples_split":randint(2,11),
         "min_samples_leaf":randint(1,6),
         "max_features":["sqrt","log2",None]},
        n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    results = {
        "logistic_regression_grid": {
            "best_params": grid.best_params_,
            "cv_roc_auc": grid.best_score_,
            "test_metrics": metrics(grid.best_estimator_, X_test, y_test)
        },
        "random_forest_randomized": {
            "best_params": rf.best_params_,
            "cv_roc_auc": rf.best_score_,
            "test_metrics": metrics(rf.best_estimator_, X_test, y_test)
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__
        }
    }
    with open("results/optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
