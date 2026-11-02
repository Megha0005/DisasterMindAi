
import threading
from pathlib import Path

import joblib

from models.count_generator.increment_generator import IncrementGenerator


class CountGeneratorService:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        model_path = (
            project_root
            / "models"
            / "count_generator"
            / "v3"
            / "increment_generator.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"V3 model not found: {model_path}"
            )

        self.model: IncrementGenerator = joblib.load(model_path)

        if not self.model.fitted:
            raise RuntimeError("Loaded V3 generator is not fitted.")

        self.lock = threading.Lock()

    def generate(self, n_samples: int):
        if not 1 <= n_samples <= 10000:
            raise ValueError(
                "n_samples must be between 1 and 10000."
            )

        with self.lock:
            generated = self.model.sample(n_samples)

        records = generated.to_dict(orient="records")

        for record in records:
            recent = record["past_1m_count"]
            previous = record["past_3m_count"]
            older = record["past_6m_count"]

            if not (0 <= recent <= previous <= older):
                raise RuntimeError(
                    "Generated count hierarchy is invalid."
                )

        return records


count_generator_service = CountGeneratorService()
