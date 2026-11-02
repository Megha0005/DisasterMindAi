
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


TEST_FILE = Path("data/processed/splits/test.csv")
OUTPUT_DIR = Path("evaluation/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features(df):
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


def evaluate_model(name, model, threshold, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "model_name": name,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }, probabilities, predictions


def main():
    print("Loading untouched test dataset...")

    test_df = pd.read_csv(TEST_FILE)
    X_test = prepare_features(test_df)
    y_test = test_df["target_next_month"].astype(int)

    print(f"Test rows: {len(X_test):,}")
    print(f"Positive cases: {y_test.sum():,}")
    print(f"Negative cases: {(y_test == 0).sum():,}")

    models = [
        {
            "name": "Logistic Regression",
            "path": Path("models/baseline/logistic_regression.joblib"),
            "config": Path("models/baseline/model_config.json")
        },
        {
            "name": "HistGradientBoosting",
            "path": Path("models/baseline/hist_gradient_boosting.joblib"),
            "config": Path("models/baseline/hist_gradient_boosting_config.json")
        }
    ]

    results = []
    prediction_data = pd.DataFrame({
        "actual": y_test.to_numpy()
    })

    for item in models:
        print(f"\nEvaluating {item['name']}...")

        model = joblib.load(item["path"])

        with open(item["config"], "r", encoding="utf-8") as file:
            config = json.load(file)

        threshold = config["classification_threshold"]

        metrics, probabilities, predictions = evaluate_model(
            item["name"],
            model,
            threshold,
            X_test,
            y_test
        )

        results.append(metrics)

        column_name = (
            "logistic_regression"
            if item["name"] == "Logistic Regression"
            else "hist_gradient_boosting"
        )

        prediction_data[f"{column_name}_probability"] = probabilities
        prediction_data[f"{column_name}_prediction"] = predictions

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison_test.csv",
        index=False
    )

    prediction_data.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "final_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n========== FINAL TEST COMPARISON ==========")
    print(
        results_df[
            [
                "model_name",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1_score"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved in:", OUTPUT_DIR)
    print("Final evaluation completed!")


if __name__ == "__main__":
    main()
