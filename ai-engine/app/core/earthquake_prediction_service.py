
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


class EarthquakePredictionService:

    def __init__(self):
        root = Path(__file__).resolve().parents[2]

        model_path = (
            root
            / "models"
            / "baseline"
            / "hist_gradient_boosting.joblib"
        )

        config_path = (
            root
            / "models"
            / "baseline"
            / "hist_gradient_boosting_config.json"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Prediction model not found: {model_path}"
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: {config_path}"
            )

        # Load the trained model.
        self.model = joblib.load(model_path)

        # Load model configuration.
        with open(config_path, "r") as file:
            self.config = json.load(file)

        self.threshold = float(
            self.config["classification_threshold"]
        )

        self.grid_resolution = int(
            self.config["grid_resolution_degrees"]
        )

        # Exact feature order used during training.
        self.features = [
            "grid_lat",
            "grid_lon",
            "year",
            "month_sin",
            "month_cos",
            "past_1m_count",
            "past_3m_count",
            "past_6m_count",
        ]

        # Verify model compatibility.
        if list(self.model.feature_names_in_) != self.features:
            raise RuntimeError(
                "Model feature order does not match training configuration."
            )

        if self.model.n_features_in_ != len(self.features):
            raise RuntimeError(
                "Model input feature count mismatch."
            )

        print("Earthquake Prediction Service Loaded Successfully.")
        print("Model:", self.config["model_name"])
        print("Threshold:", self.threshold)
        print("Grid resolution:", self.grid_resolution, "degrees")

    def predict(
        self,
        latitude,
        longitude,
        year,
        month,
        past_1m_count,
        past_3m_count,
        past_6m_count,
    ):

        # Validate geographic coordinates.
        if not np.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not np.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        # Validate year and month.
        if not isinstance(year, (int, np.integer)):
            raise ValueError("Year must be an integer.")

        if not 2000 <= year <= 2100:
            raise ValueError(
                "Year must be between 2000 and 2100."
            )

        if not isinstance(month, (int, np.integer)):
            raise ValueError("Month must be an integer.")

        if not 1 <= month <= 12:
            raise ValueError(
                "Month must be between 1 and 12."
            )

        # Validate historical counts.
        counts = [
            past_1m_count,
            past_3m_count,
            past_6m_count,
        ]

        if any(
            not isinstance(value, (int, np.integer))
            for value in counts
        ):
            raise ValueError(
                "Historical earthquake counts must be integers."
            )

        if any(value < 0 for value in counts):
            raise ValueError(
                "Earthquake counts cannot be negative."
            )

        if not (
            past_1m_count
            <= past_3m_count
            <= past_6m_count
        ):
            raise ValueError(
                "Historical count hierarchy is invalid."
            )

        # Convert GPS coordinates to the dataset's
        # 2-degree grid-cell lower-boundary coordinates.
        grid_lat = float(
            np.floor(latitude / self.grid_resolution)
            * self.grid_resolution
        )

        grid_lon = float(
            np.floor(longitude / self.grid_resolution)
            * self.grid_resolution
        )

        # Prepare features using the exact training transformations.
        features = pd.DataFrame(
            [{
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "year": year,

                "month_sin": np.sin(
                    2 * np.pi * month / 12
                ),

                "month_cos": np.cos(
                    2 * np.pi * month / 12
                ),

                "past_1m_count": np.log1p(
                    past_1m_count
                ),

                "past_3m_count": np.log1p(
                    past_3m_count
                ),

                "past_6m_count": np.log1p(
                    past_6m_count
                ),
            }],
            columns=self.features
        )

        # Generate occurrence probability.
        probability = float(
            self.model.predict_proba(features)[0, 1]
        )

        # Apply the validation-selected threshold.
        prediction = int(
            probability >= self.threshold
        )

        return {
            "model": self.config["model_name"],

            "prediction_horizon": (
                "Next calendar month"
            ),

            "probability": probability,

            "classification_threshold": (
                self.threshold
            ),

            "prediction": prediction,

            "prediction_label": (
                "Earthquake occurrence indicated"
                if prediction == 1
                else "Earthquake occurrence not indicated"
            ),

            "grid_cell": {
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "latitude_range": [
                    grid_lat,
                    grid_lat + self.grid_resolution
                ],
                "longitude_range": [
                    grid_lon,
                    grid_lon + self.grid_resolution
                ],
                "resolution_degrees": self.grid_resolution
            },

            "disclaimer": (
                "Retrospective statistical model for "
                "next-calendar-month M4+ earthquake "
                "occurrence in a 2-degree grid cell. "
                "Not an operational earthquake early-warning "
                "system or a precise earthquake prediction."
            )
        }


earthquake_prediction_service = EarthquakePredictionService()
