
from fastapi import FastAPI
from datetime import datetime

from app.api.count_generator import (
    router as count_generator_router
)

from app.api.earthquake_prediction import (
    router as earthquake_prediction_router
)

from app.api.disaster_scenario import (
    router as disaster_scenario_router
)


app = FastAPI(
    title="DisasterMind AI",
    description=(
        "AI-powered Natural Disaster Prediction "
        "and Scenario Generation Platform"
    ),
    version="1.0.0"
)


# Register API routers
app.include_router(earthquake_prediction_router)
app.include_router(count_generator_router)
app.include_router(disaster_scenario_router)


# Root endpoint
@app.get("/")
async def root():
    return {
        "project": "DisasterMind AI",
        "message": "AI Engine is running successfully!",
        "status": "online"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DisasterMind AI Engine",
        "timestamp": datetime.now().isoformat()
    }


# GPU status endpoint
@app.get("/gpu")
async def gpu_status():
    import torch

    return {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "No GPU detected"
        )
    }
