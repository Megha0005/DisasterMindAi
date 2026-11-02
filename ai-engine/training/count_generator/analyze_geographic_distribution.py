
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error
)

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

def prepare(df):
    result = df.copy()

    result["recent"] = result["past_1m_count"]
    result["previous"] = (
        result["past_3m_count"] - result["past_1m_count"]
    )
    result["older"] = (
        result["past_6m_count"] - result["past_3m_count"]
    )

    return result

train = prepare(train)
validation = prepare(validation)

features = ["grid_lat", "grid_lon"]

for target in ["recent", "previous", "older"]:

    print(f"\n========== {target.upper()} ==========")

    y_train = train[target]
    y_val = validation[target]

    # Stage 1: Probability of a non-zero count
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(train[features], (y_train > 0).astype(int))

    probabilities = classifier.predict_proba(
        validation[features]
    )[:, 1]

    auc = roc_auc_score((y_val > 0).astype(int), probabilities)
    pr_auc = average_precision_score(
        (y_val > 0).astype(int), probabilities
    )

    print("Occurrence ROC-AUC:", round(auc, 4))
    print("Occurrence PR-AUC:", round(pr_auc, 4))

    # Stage 2: Positive count magnitude
    positive_train = train[target] > 0

    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(
        train.loc[positive_train, features],
        np.log1p(y_train[positive_train])
    )

    positive_val = y_val > 0

    predictions = np.expm1(
        regressor.predict(validation.loc[positive_val, features])
    )

    mae = mean_absolute_error(
        y_val[positive_val],
        predictions
    )

    print("Positive-count MAE:", round(mae, 4))

