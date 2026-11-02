
from fastapi import APIRouter, HTTPException

from app.core.earthquake_prediction_service import (
    earthquake_prediction_service
)

from app.schemas.earthquake_prediction import (
    EarthquakePredictionRequest,
    EarthquakePredictionResponse
)


router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Earthquake Prediction"]
)


@router.post(
    "/earthquake",
    response_model=EarthquakePredictionResponse
)
async def predict_earthquake(
    request: EarthquakePredictionRequest
):

    try:
        result = earthquake_prediction_service.predict(
            latitude=request.latitude,
            longitude=request.longitude,
            year=request.year,
            month=request.month,
            past_1m_count=request.past_1m_count,
            past_3m_count=request.past_3m_count,
            past_6m_count=request.past_6m_count
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
            detail="Earthquake prediction failed."
        ) from error
