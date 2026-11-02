from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class DisasterScenarioRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    n_scenarios: int = Field(default=5, ge=1, le=1000)
    scenario_name: str = Field(
        default="Earthquake Scenario",
        min_length=3,
        max_length=100
    )


class ScenarioRecord(BaseModel):
    scenario_id: int
    scenario_type: Literal["synthetic_earthquake_count"]
    grid_lat: float
    grid_lon: float
    latitude_range: list[float]
    longitude_range: list[float]
    past_1m_count: int
    past_3m_count: int
    past_6m_count: int


class DisasterScenarioResponse(BaseModel):
    model: str
    scenario_name: str
    generation_method: str
    generated_scenarios: int
    scenarios: list[ScenarioRecord]
    disclaimer: str
