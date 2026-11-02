
from fastapi import APIRouter, HTTPException

from app.core.disaster_scenario_service import (
    disaster_scenario_service
)

from app.schemas.disaster_scenario import (
    DisasterScenarioRequest,
    DisasterScenarioResponse
)


router = APIRouter(
    prefix="/api/v1/scenarios",
    tags=["Disaster Scenario Generation"]
)


@router.post(
    "/generate",
    response_model=DisasterScenarioResponse
)
async def generate_disaster_scenarios(
    request: DisasterScenarioRequest
):

    try:
        result = disaster_scenario_service.generate(
            latitude=request.latitude,
            longitude=request.longitude,
            n_scenarios=request.n_scenarios,
            scenario_name=request.scenario_name
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Disaster scenario generation failed."
        ) from error
