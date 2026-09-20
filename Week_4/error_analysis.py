"""
Week 4 - Error Analysis

Identifies incorrect predictions and summarizes the errors by actual and
predicted class.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    error_mask = y_test != predictions

    errors = pd.DataFrame(
        {
            "actual": y_test[error_mask],
            "predicted": predictions[error_mask],
        }
    )

    print("=== Error Analysis ===")
    print(f"Total test samples: {len(y_test)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Error rate: {error_mask.mean():.4f}")

    print("\nIncorrect Predictions:")
    print(errors.to_string(index=False))

    print("\nError Summary by Actual/Predicted Class:")
    if not errors.empty:
        print(
            errors.groupby(["actual", "predicted"])
            .size()
            .rename("count")
            .reset_index()
            .to_string(index=False)
        )
    else:
        print("No classification errors in this test split.")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
