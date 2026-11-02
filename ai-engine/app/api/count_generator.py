
from fastapi import APIRouter, HTTPException

from app.core.count_generator_service import count_generator_service
from app.schemas.count_generator import (
    CountGenerationRequest,
    CountGenerationResponse
)

router = APIRouter(
    prefix="/api/v1/generator",
    tags=["Synthetic Data Generation"]
)


@router.post(
    "/counts",
    response_model=CountGenerationResponse
)
async def generate_counts(request: CountGenerationRequest):

    try:
        records = count_generator_service.generate(
            request.n_samples
        )

        return CountGenerationResponse(
            model="V3",
            generated_samples=len(records),
            records=records
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Count generation failed."
        ) from error
