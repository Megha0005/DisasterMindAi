
import json
import joblib
import pandas as pd

from pathlib import Path
from models.count_generator.increment_generator import IncrementGenerator


def main():

    root = Path(__file__).resolve().parents[2]

    data_path = root / "data/processed/count_generator_train.csv"
    model_dir = root / "models/count_generator/v3"

    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    df = pd.read_csv(data_path)

    print("Training Increment Generator V3...")

    model = IncrementGenerator(random_state=42)
    model.fit(df)

    print("\nGenerating verification samples...")
    synthetic = model.sample(10000)

    assert (
        synthetic["past_1m_count"] <= synthetic["past_3m_count"]
    ).all()

    assert (
        synthetic["past_3m_count"] <= synthetic["past_6m_count"]
    ).all()

    print("Generated samples:", len(synthetic))
    print("Count hierarchy validation: PASSED")

    model_path = model_dir / "increment_generator.joblib"

    joblib.dump(model, model_path)

    metadata = {
        "model": "IncrementGenerator",
        "version": "V3",
        "random_state": 42,
        "training_rows": len(df),
        "unique_increment_tuples": len(model.increment_tuples),
        "verification_samples": len(synthetic)
    }

    metadata_path = model_dir / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining completed successfully!")
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    main()
