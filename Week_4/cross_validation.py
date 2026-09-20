"""
Week 4 - Stratified K-Fold Cross-Validation
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    print("=== 5-Fold Stratified Cross-Validation ===")

    for metric in scoring:
        test_scores = results[f"test_{metric}"]
        train_scores = results[f"train_{metric}"]

        print(
            f"{metric:10s} | "
            f"test mean={test_scores.mean():.4f}, "
            f"test std={test_scores.std():.4f}, "
            f"train mean={train_scores.mean():.4f}"
        )


if __name__ == "__main__":
    main()
