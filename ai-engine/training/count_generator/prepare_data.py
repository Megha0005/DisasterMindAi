import os
import pandas as pd
import numpy as np

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "splits", "train.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "count_generator_train.csv"
)

# Load training data
print("Loading training dataset...")
df = pd.read_csv(INPUT_PATH)

# Select only required features
FEATURE_COLUMNS = [
    "grid_lat",
    "grid_lon",
    "year",
    "month",
    "earthquake_count",
    "past_1m_count",
    "past_3m_count",
    "past_6m_count"
]

data = df[FEATURE_COLUMNS].copy()

# Validate missing values
print("\nChecking missing values...")
print(data.isnull().sum())

# Validate count constraints
count_columns = [
    "earthquake_count",
    "past_1m_count",
    "past_3m_count",
    "past_6m_count"
]

assert not data[count_columns].isnull().any().any()
assert (data[count_columns] >= 0).all().all()

assert (
    data["past_1m_count"] <= data["past_3m_count"]
).all()

assert (
    data["past_3m_count"] <= data["past_6m_count"]
).all()

# Validate geographical boundaries
assert data["grid_lat"].between(-90, 90).all()
assert data["grid_lon"].between(-180, 180).all()

# Validate calendar
assert data["month"].between(1, 12).all()

# Save prepared dataset
data.to_csv(OUTPUT_PATH, index=False)

print("\nData preparation completed!")
print("Rows:", len(data))
print("Columns:", len(data.columns))
print("Output:", OUTPUT_PATH)