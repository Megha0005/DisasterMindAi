# DisasterMind AI

Full-stack natural-disaster risk analysis and scenario-generation project.

## Architecture
- `client/`: React + Vite frontend
- `server/`: Node.js + Express API and MongoDB integration
- `ai-engine/`: Python + FastAPI deep-learning service
- `docs/`: project documentation

## Prerequisites
- Node.js (LTS)
- Python 3.11 or compatible version supported by your PyTorch installation
- MongoDB local instance or MongoDB Atlas account

## Run locally

### 1. Frontend
```bash
cd client
npm install
npm run dev
```

### 2. Express backend
```bash
cd server
npm install
# Copy .env.example to .env and configure values
npm run dev
```

### 3. Python AI engine
```bash
cd ai-engine
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The starter health endpoints are `/api/health` (Express) and `/health` (FastAPI).

## Important
This repository is a starter scaffold. Model implementations, dataset ingestion, authentication, and production security still need to be developed and validated. Do not use its outputs for real-world emergency warnings.
