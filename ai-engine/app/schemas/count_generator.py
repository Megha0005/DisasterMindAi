
from pydantic import BaseModel, Field, ConfigDict
from typing import List


class CountGenerationRequest(BaseModel):
    n_samples: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of synthetic count records to generate."
    )


class GeneratedCountRecord(BaseModel):
    past_1m_count: int
    past_3m_count: int
    past_6m_count: int


class CountGenerationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "V3",
                "generated_samples": 2,
                "records": [
                    {
                        "past_1m_count": 1,
                        "past_3m_count": 2,
                        "past_6m_count": 4
                    },
                    {
                        "past_1m_count": 0,
                        "past_3m_count": 1,
                        "past_6m_count": 3
                    }
                ]
            }
        }
    )

    model: str
    generated_samples: int
    records: List[GeneratedCountRecord]
