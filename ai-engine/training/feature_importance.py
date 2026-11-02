
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


MODEL_FILE = Path("models/baseline/hist_gradient_boosting.joblib")
VAL_FILE = Path("data/processed/splits/validation.csv")
OUTPUT_FILE = Path("evaluation/baseline/feature_importance.csv")

RANDOM_STATE = 42
SAMPLE_SIZE = 40000


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


def main():
    print("Loading model and validation data...")

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(VAL_FILE)

    # Use a reproducible validation sample to reduce computation time.
    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    X = prepare_features(df)
    y = df["target_next_month"].astype(int)

    print(f"Validation sample: {len(X):,} rows")
    print(f"Features: {len(X.columns)}")

    baseline_auc = roc_auc_score(
        y,
        model.predict_proba(X)[:, 1]
    )

    print(f"\nBaseline ROC-AUC: {baseline_auc:.4f}")
    print("\nCalculating permutation importance...")

    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== FEATURE IMPORTANCE ==========")

    for _, row in importance_df.iterrows():
        print(
            f"{row['feature']:<20} "
            f"{row['importance_mean']:.6f} "
            f"+/- {row['importance_std']:.6f}"
        )

    print("\nSaved to:", OUTPUT_FILE)
    print("\nFeature importance analysis completed!")


if __name__ == "__main__":
    main()
