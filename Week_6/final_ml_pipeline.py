"""Week 6 Final ML Project - Reproducible end-to-end experiment."""
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

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 6),
        "precision": round(precision_score(y_test, pred), 6),
        "recall": round(recall_score(y_test, pred), 6),
        "f1": round(f1_score(y_test, pred), 6),
        "roc_auc": round(roc_auc_score(y_test, prob), 6),
    }

def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    baseline_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])
    baseline_lr.fit(X_train, y_train)

    tuned_lr = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        {"model__C": [0.01, 0.1, 1, 10, 100],
         "model__solver": ["liblinear", "lbfgs"]},
        scoring="roc_auc", cv=cv, n_jobs=-1
    )
    tuned_lr.fit(X_train, y_train)

    baseline_rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    baseline_rf.fit(X_train, y_train)

    tuned_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": randint(100, 501),
         "max_depth": [None, 5, 10, 15, 20, 25],
         "min_samples_split": randint(2, 11),
         "min_samples_leaf": randint(1, 6),
         "max_features": ["sqrt", "log2", None]},
        n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    tuned_rf.fit(X_train, y_train)

    results = {
        "dataset": {"name": "Breast Cancer Wisconsin (Diagnostic)", "samples": len(X), "features": X.shape[1]},
        "split": {"test_size": 0.20, "stratified": True, "random_state": RANDOM_STATE},
        "validation": {"method": "5-fold Stratified Cross-Validation", "primary_metric": "ROC-AUC"},
        "models": {
            "logistic_regression_baseline": evaluate(baseline_lr, X_test, y_test),
            "logistic_regression_grid_search": {
                "best_params": tuned_lr.best_params_,
                "cv_roc_auc": round(tuned_lr.best_score_, 6),
                "test_metrics": evaluate(tuned_lr.best_estimator_, X_test, y_test),
            },
            "random_forest_baseline": evaluate(baseline_rf, X_test, y_test),
            "random_forest_randomized_search": {
                "best_params": tuned_rf.best_params_,
                "cv_roc_auc": round(tuned_rf.best_score_, 6),
                "test_metrics": evaluate(tuned_rf.best_estimator_, X_test, y_test),
            },
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    }
    with open("results/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
