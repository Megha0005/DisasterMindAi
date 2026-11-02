
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="DisasterMind AI",
    description="AI-powered Natural Disaster Prediction and Scenario Generation Platform",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "project": "DisasterMind AI",
        "message": "AI Engine is running successfully!",
        "status": "online"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DisasterMind AI Engine",
        "timestamp": datetime.now().isoformat()
    }


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
