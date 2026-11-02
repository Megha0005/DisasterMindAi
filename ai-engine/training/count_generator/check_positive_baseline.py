
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]

train = pd.read_csv(
    ROOT / "data/processed/count_generator_train.csv"
)

validation = pd.read_csv(
    ROOT / "data/processed/count_generator_validation.csv"
)

for df in [train, validation]:
    df["recent"] = df["past_1m_count"]
    df["previous"] = df["past_3m_count"] - df["past_1m_count"]
    df["older"] = df["past_6m_count"] - df["past_3m_count"]

for target in ["recent", "previous", "older"]:

    train_positive = train.loc[train[target] > 0, target]
    val_positive = validation.loc[validation[target] > 0, target]

    baseline = train_positive.mean()

    predictions = np.full(len(val_positive), baseline)

    mae = mean_absolute_error(val_positive, predictions)

    print(f"\n{target.upper()}")
    print("Training positive mean:", round(baseline, 4))
    print("Baseline positive-count MAE:", round(mae, 4))

