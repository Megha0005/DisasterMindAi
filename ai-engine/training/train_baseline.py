
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("evaluation/baseline")

FEATURES = [
    "grid_lat",
    "grid_lon",
    "year",
    "month_sin",
    "month_cos",
    "past_1m_count",
    "past_3m_count",
    "past_6m_count",
]

TARGET = "target_next_month"


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def prepare_features(df):
    """Create leakage-safe model inputs."""
    X = df[
        [
            "grid_lat",
            "grid_lon",
            "year",
            "month",
            "past_1m_count",
            "past_3m_count",
            "past_6m_count",
        ]
    ].copy()

    # Represent month as a cyclic feature.
    X["month_sin"] = np.sin(
        2 * np.pi * X["month"] / 12
    )

    X["month_cos"] = np.cos(
        2 * np.pi * X["month"] / 12
    )

    # Reduce skew in earthquake activity counts.
    for column in [
        "past_1m_count",
        "past_3m_count",
        "past_6m_count",
    ]:
        X[column] = np.log1p(X[column])

    return X[FEATURES]


def evaluate_model(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n========== {name} ==========")

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_true, probabilities), 4)
    )

    print(
        "PR-AUC:",
        round(average_precision_score(y_true, probabilities), 4)
    )

    print(
        "Precision:",
        round(precision_score(y_true, predictions, zero_division=0), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_true, predictions, zero_division=0), 4)
    )

    print(
        "F1-score:",
        round(f1_score(y_true, predictions, zero_division=0), 4)
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )


def main():
    print("Loading chronological datasets...")

    train_df = load_dataset("train.csv")
    validation_df = load_dataset("validation.csv")

    X_train = prepare_features(train_df)
    y_train = train_df[TARGET].astype(int)

    X_validation = prepare_features(validation_df)
    y_validation = validation_df[TARGET].astype(int)

    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_validation):,}")
    print(f"Input features: {len(FEATURES)}")

    # Historical-activity benchmark.
    baseline_probabilities = (
        validation_df["past_1m_count"].to_numpy() > 0
    ).astype(float)

    evaluate_model(
        "Historical Activity Baseline",
        y_validation,
        baseline_probabilities
    )

    # Logistic Regression pipeline.
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    evaluate_model(
        "Logistic Regression",
        y_validation,
        validation_probabilities
    )

    # Save trained model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "logistic_regression.joblib"
    )

    # Save validation predictions for later analysis.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        "month_period": validation_df["month_period"],
        "grid_lat": validation_df["grid_lat"],
        "grid_lon": validation_df["grid_lon"],
        "actual": y_validation,
        "predicted_probability": validation_probabilities,
    })

    predictions_df.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False
    )

    print("\n========== TRAINING COMPLETE ==========")
    print(
        "Model saved:",
        (MODEL_DIR / "logistic_regression.joblib").resolve()
    )

    print(
        "Predictions saved:",
        (REPORT_DIR / "validation_predictions.csv").resolve()
    )


if __name__ == "__main__":
    main()
