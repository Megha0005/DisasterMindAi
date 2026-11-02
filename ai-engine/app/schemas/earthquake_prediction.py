
from pydantic import BaseModel, Field, ConfigDict


class EarthquakePredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    past_1m_count: int = Field(ge=0)
    past_3m_count: int = Field(ge=0)
    past_6m_count: int = Field(ge=0)


class GridCellInfo(BaseModel):
    grid_lat: float
    grid_lon: float
    latitude_range: list[float]
    longitude_range: list[float]
    resolution_degrees: int


class EarthquakePredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "HistGradientBoostingClassifier",
                "prediction_horizon": "Next calendar month",
                "probability": 0.738,
                "classification_threshold": 0.8086,
                "prediction": 0,
                "prediction_label": "Earthquake occurrence not indicated",
                "grid_cell": {
                    "grid_lat": 20.0,
                    "grid_lon": 78.0,
                    "latitude_range": [20.0, 22.0],
                    "longitude_range": [78.0, 80.0],
                    "resolution_degrees": 2
                },
                "disclaimer": "Retrospective statistical model."
            }
        }
    )

    model: str
    prediction_horizon: str
    probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    prediction_label: str
    grid_cell: GridCellInfo
    disclaimer: str
