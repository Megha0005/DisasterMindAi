
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight


# Paths
TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/validation.csv")

MODEL_DIR = Path("models/baseline")
EVAL_DIR = Path("evaluation/baseline")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df):
    """Build the same features used by the Logistic Regression baseline."""

    month_period = pd.PeriodIndex(df["month_period"], freq="M")

    X = pd.DataFrame(index=df.index)

    X["grid_lat"] = df["grid_lat"]
    X["grid_lon"] = df["grid_lon"]
    X["year"] = month_period.year
    X["month_sin"] = np.sin(2 * np.pi * month_period.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month_period.month / 12)

    X["past_1m_count"] = np.log1p(df["past_1m_count"])
    X["past_3m_count"] = np.log1p(df["past_3m_count"])
    X["past_6m_count"] = np.log1p(df["past_6m_count"])

    return X.astype("float32")


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
    }


def main():
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    y_train = train_df["target_next_month"].astype(int)
    y_val = val_df["target_next_month"].astype(int)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {list(X_train.columns)}")

    # Balance the classes without changing the original dataset.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print("\nTraining HistGradientBoosting...")

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print(f"Iterations used: {model.n_iter_}")

    print("\nGenerating validation predictions...")
    probabilities = model.predict_proba(X_val)[:, 1]

    # Metrics at the default threshold.
    default_metrics = calculate_metrics(
        y_val,
        probabilities,
        0.5
    )

    # Find the threshold that maximizes validation F1.
    precision, recall, thresholds = precision_recall_curve(
        y_val,
        probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    best_metrics = calculate_metrics(
        y_val,
        probabilities,
        best_threshold
    )

    # Save model.
    model_path = MODEL_DIR / "hist_gradient_boosting.joblib"
    joblib.dump(model, model_path)

    # Save validation predictions.
    predictions_df = pd.DataFrame({
        "actual": y_val.to_numpy(),
        "probability": probabilities,
        "prediction_at_0_5": (probabilities >= 0.5).astype(int),
        "prediction_at_best_threshold": (
            probabilities >= best_threshold
        ).astype(int)
    })

    predictions_df.to_csv(
        EVAL_DIR / "hist_gradient_boosting_predictions.csv",
        index=False
    )

    # Save metrics.
    results = {
        "model_name": "HistGradientBoosting",
        "iterations_used": int(model.n_iter_),
        "default_threshold_metrics": default_metrics,
        "best_validation_threshold_metrics": best_metrics
    }

    with open(
        EVAL_DIR / "hist_gradient_boosting_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n========== HIST GRADIENT BOOSTING RESULTS ==========")

    print("\nAt threshold 0.5:")
    for metric, value in default_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nAt best validation F1 threshold:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSaved model:", model_path)
    print("Saved metrics:", EVAL_DIR / "hist_gradient_boosting_metrics.json")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
