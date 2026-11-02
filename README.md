# DisasterMind AI

### AI-Powered Natural Disaster Risk Analysis, Prediction and Scenario Generation

## 1. About the Project

DisasterMind AI is a full-stack, AI-powered disaster analysis platform that uses Machine Learning, Deep Learning, historical earthquake data, statistical modelling, and interactive visualization.

The project focuses on analysing historical earthquake activity and generating analytical insights through a web application.

It provides:

- Earthquake occurrence classification.

- Historical earthquake data analysis.

- Geographical risk visualization.

- Synthetic earthquake-count scenario generation.

- Deep Learning-based anomaly exploration.

- Interactive dashboards and reports.

- User registration and authentication.

**Important:** This is a research-oriented analytical prototype. It does not predict the exact time, location, or magnitude of earthquakes. It is not an operational earthquake early-warning system.

---

## 2. Problem Statement

Natural disasters can cause significant damage to human life, infrastructure, and the economy.

Some major challenges include:

1. Large volumes of historical earthquake data are difficult to analyse manually.

2. Earthquake activity varies across geographical locations and time.

3. Raw geological datasets are difficult for ordinary users to understand.

4. Historical patterns cannot guarantee future earthquake events.

5. Researchers need tools to explore different possible disaster scenarios.

6. Data processing, Machine Learning, and visualization are often performed using separate tools.

## 3. Our Proposed Solution

DisasterMind AI provides a unified platform for historical earthquake analysis and scenario exploration.

### How We Solved the Problem

1. Data Collection: Retrieved historical earthquake records using the USGS Earthquake API.

2. Data Cleaning: Removed invalid and incomplete records and prepared structured datasets.

3. Feature Engineering: Converted earthquake records into geographical grid-cell and monthly observations.

4. Machine Learning: Trained models to classify next-calendar-month M4+ earthquake occurrence in geographical grid cells.

5. Deep Learning: Explored Autoencoders and Variational Autoencoders for anomaly analysis.

6. Scenario Generation: Developed a statistical generator to produce synthetic earthquake-count scenarios.

7. Visualization: Created interactive maps, charts, dashboards, and reports.

8. Integration: Connected the React frontend, Node.js backend, Python AI engine, and MongoDB database.

---

## 4. System Architecture

```text

USER

|

v

REACT + VITE

PORT 5173

|

v

NODE.JS + EXPRESS

PORT 5000

|

+--------+--------+

|                 |

v                 v

MONGODB ATLAS       PYTHON FASTAPI

USER DATABASE       AI ENGINE

PORT 8000

|

+-----------+-----------+

|           |           |

v           v           v

MACHINE    DEEP        SCENARIO

LEARNING   LEARNING    GENERATOR

|

v

TRAINED MODELS

AND DATASETS

```

### Architecture Components

| Component     | Responsibility              |

| ------------- | --------------------------- |

| React + Vite  | Frontend and user interface |

| Node.js       | Backend runtime             |

| Express.js    | REST APIs and gateway       |

| MongoDB Atlas | User database               |

| Python        | AI and data processing      |

| FastAPI       | AI service APIs             |

| Scikit-learn  | Machine Learning            |

| PyTorch       | Deep Learning               |

## 5. Technologies Used and Why

### Frontend

React.js

- Component-based frontend development.

- Reusable UI components.

- Dynamic and interactive pages.

- Used for dashboard, login, registration, maps, and analysis pages.

- Selected over plain HTML/CSS/JS because the application has multiple interactive pages and reusable components.

Vite

- Fast development server and build tool.

- Used to run and build the React application.

- Selected for its fast development experience.

React Router

- Handles navigation between pages.

- Allows client-side routing without full-page reloads.

Axios

- Handles HTTP requests.

- Connects the frontend with backend APIs.

React Leaflet and Leaflet

- Used for interactive geographical maps.

- Helps visualize geographical earthquake information and grid cells.

Recharts

- Used for charts and data visualization.

- Helps display statistical and analytical results.

Lucide React

- Provides consistent icons for navigation, buttons, and dashboard components.

### Backend

Node.js

- JavaScript runtime.

- Handles application-level backend operations and API requests.

Express.js

- Lightweight backend framework.

- Used to create REST APIs, authentication routes, and AI gateway routes.

MongoDB Atlas

- Cloud-hosted NoSQL database.

- Stores registered user information.

- Selected because the current application needs flexible document-based user storage.

Mongoose

- MongoDB object modelling.

- Defines schemas and validates user information.

bcryptjs

- Hashes passwords before storing them.

- Helps protect user credentials.

JSON Web Token (JWT)

- Generates authentication tokens.

- Supports token-based user authentication.

dotenv

- Loads environment variables from the backend `.env` file.

CORS

- Helps configure cross-origin communication between frontend and backend.

### AI and Machine Learning

Python

- Main language for data science and AI.

- Used for data processing, training, evaluation, and inference.

FastAPI

- Exposes Python AI functionality through REST APIs.

- Provides automatic API documentation.

- Selected to keep AI services separate from the Node.js backend.

NumPy

- Numerical computation and array operations.

Pandas

- Data cleaning, filtering, aggregation, and feature engineering.

SciPy

- Scientific and statistical computation.

Scikit-learn

- Classification algorithms, preprocessing, and model evaluation.

PyTorch

- Deep Learning model development and GPU acceleration.

Joblib

- Saves and loads trained Machine Learning models.

Pydantic

- Validates API request and response data.

Matplotlib

- Static scientific visualization.

Plotly

- Interactive charts and visualization.

pgmpy

- Used for probabilistic graphical modelling experiments.

### Why Python for AI and Node.js for Backend?

Python has a mature ecosystem for Machine Learning, Deep Learning, and scientific computing.

Node.js is suitable for application APIs and frontend communication.

Separating these responsibilities makes the application modular and easier to maintain.

## 6. Dataset and Data Processing

### Data Source

USGS Earthquake Catalog / FDSN Event API:

[https://earthquake.usgs.gov/fdsnws/event/1/](https://earthquake.usgs.gov/fdsnws/event/1/)

Historical earthquake data from 2000 to 2025 was used, with a target magnitude threshold of M4+.

### Dataset Summary

| Description                    | Count     |

| ------------------------------ | --------- |

| Raw earthquake records         | 357,841   |

| Cleaned records                | 357,840   |

| Geographical grid cells        | 4,352     |

| Grid-cell monthly observations | 1,318,656 |

| Negative observations          | 90.28%    |

| Positive observations          | 9.72%     |

### Processing Steps

1. Download historical earthquake records.

2. Clean invalid and incomplete records.

3. Apply the M4+ magnitude filter.

4. Process latitude and longitude.

5. Divide geographical areas into 2° × 2° grid cells.

6. Aggregate earthquake activity by month.

7. Generate spatiotemporal features.

8. Create the next-calendar-month occurrence target.

9. Split the data chronologically.

### Dataset Split

| Dataset    | Period                        |

| ---------- | ----------------------------- |

| Training   | July 2000 – December 2019     |

| Validation | January 2020 – December 2022  |

| Testing    | January 2023 – September 2025 |

Chronological splitting helps evaluate performance on later periods.

## 7. Machine Learning and Deep Learning Models

### 7.1 HistGradientBoosting Classifier

Purpose: Main earthquake occurrence classification model.

It uses an ensemble of decision trees to learn nonlinear relationships in structured data.

Why selected:

- Handles nonlinear patterns.

- Works well with tabular datasets.

- Captures feature interactions.

- Achieved the best classification performance in our experiments.

It estimates whether a geographical 2° × 2° grid cell will experience at least one M4+ earthquake during the next calendar month.

Test Results:

| Metric    | Result   |

| --------- | -------- |

| ROC-AUC   | 0.898668 |

| PR-AUC    | 0.654764 |

| Precision | 0.620974 |

| Recall    | 0.585353 |

| F1-score  | 0.602638 |

| Threshold | 0.8086   |

Confusion Matrix:

|                 | Predicted Negative | Predicted Positive |

| --------------- | ------------------ | ------------------ |

| Actual Negative | 123007             | 5425               |

| Actual Positive | 6296               | 8888               |

### 7.2 Logistic Regression

Purpose: Baseline classification model.

Why selected:

- Simple and interpretable.

- Provides a baseline for comparison.

- Helps evaluate whether complex models improve performance.

Results:

| Metric   | Result   |

| -------- | -------- |

| ROC-AUC  | 0.866198 |

| PR-AUC   | 0.611981 |

| F1-score | 0.584522 |

### 7.3 Multi-Layer Perceptron (MLP)

Purpose: Neural-network-based classification experiment.

Why selected:

- Learns nonlinear relationships.

- Provides comparison with traditional Machine Learning models.

Results:

| Metric   | Result |

| -------- | ------ |

| ROC-AUC  | 0.8791 |

| PR-AUC   | 0.6388 |

| F1-score | 0.5923 |

### 7.4 Autoencoder

Purpose: Reconstruction-based anomaly exploration.

Working:

1. Input data enters the encoder.

2. Encoder extracts important features.

3. Features are compressed into a latent representation.

4. Decoder reconstructs the input.

5. Reconstructed output is compared with original input.

6. Reconstruction error is calculated.

7. High reconstruction error can indicate an unusual observation.

Used for descriptive anomaly analysis, not earthquake forecasting.

### 7.5 Variational Autoencoder (VAE)

Purpose: Probabilistic latent representation and anomaly exploration.

Why selected:

- Learns a probabilistic latent representation.

- Supports reconstruction-based analysis.

- Helps explore unusual data patterns.

It is not used as an operational earthquake predictor.

### 7.6 Restricted Boltzmann Machine (RBM)

Purpose: Experimental generative modelling.

The RBM was explored as a possible generative model, but the experiment did not produce sufficiently reliable generative results. It was not selected as the primary generator.

### 7.7 Historical Earthquake Count Generator V3

Purpose: Synthetic earthquake-count generation.

Why selected:

- Samples historical count patterns.

- Supports statistical scenario exploration.

- Provides an empirical generation approach.

V3 was selected as the primary count generator.

Limitation: It underestimates some historical means and tail behaviour. It is not a forecasting model.

### 7.8 Historical Earthquake Count Generator V4

Purpose: Experimental alternative to V3.

V4 was evaluated experimentally. V3 remained the selected primary version.

### 7.9 Scenario Generation

The scenario API combines generated counts with selected geographical grid information.

Validation results from 1,000 scenarios:

- Hierarchy violations: 0

- Negative counts: 0

- All-zero scenarios: 737

Important:

- Scenarios are synthetic, not actual future earthquakes.

- The current generator is not geographically conditioned.

- All-zero scenarios must be considered when interpreting results.

- Synthetic outputs are not calibrated future-event probabilities.

## 8. Model Evaluation and Limitations

We use:

- Precision

- Recall

- F1-score

- ROC-AUC

- PR-AUC

- Confusion Matrix

ROC-AUC alone is not sufficient because the dataset is imbalanced.

### What the model can do

- Classify next-calendar-month M4+ occurrence for a grid cell.

- Analyse historical spatiotemporal patterns.

- Support retrospective model evaluation.

- Provide analytical risk information.

### What the model cannot do

- Predict exact earthquake time.

- Predict exact epicentre.

- Predict exact future magnitude.

- Guarantee whether an earthquake will occur.

- Replace official earthquake monitoring systems.

## 9. Web Application Pages

### 9.1 Login Page

- Allows existing users to sign in.

- Accepts email and password.

- Communicates with the backend authentication API.

- Handles login errors and loading states.

- Stores authentication information after successful login.

### 9.2 Registration Page

- Allows new users to create accounts.

- Accepts name, email, password, and confirmation.

- Performs input validation.

- Uses backend registration.

- Passwords are hashed using bcryptjs.

### 9.3 Dashboard

- Main landing page.

- Displays platform overview and analytical information.

- Provides navigation to application modules.

- Includes profile dropdown and logout functionality.

### 9.4 Disaster Map

- Interactive geographical map.

- Displays geographical earthquake information.

- Supports grid-based exploration.

- Helps understand geographical patterns.

It is not a real-time early-warning map.

### 9.5 Risk Analysis

- Uses the earthquake occurrence classification service.

- Allows grid-cell analysis.

- Displays model output and relevant geographical information.

- Presents next-calendar-month M4+ occurrence classification.

It does not provide exact earthquake predictions.

### 9.6 Scenario Generator

- Sends requests to the scenario-generation API.

- Generates synthetic earthquake-count scenarios.

- Displays generated analytical outputs.

- Supports scenario exploration.

### 9.7 Probabilistic Models

- Presents probabilistic and statistical modelling concepts.

- Supports exploration of model-related analytical outputs.

- Provides a space for probabilistic analysis.

### 9.8 Deep Learning Lab

- Explores Deep Learning experiments.

- Includes Autoencoder and VAE-related analysis.

- Supports reconstruction-based anomaly exploration.

Anomaly scores are not earthquake probabilities.

### 9.9 Reports

- Presents analytical results.

- Displays statistical summaries and model-related information.

- Provides a report-oriented view of the analysis.

### 9.10 Settings

- Displays user profile information.

- Shows account and security information.

- Provides logout functionality.

## 10. Project Structure

```

DisasterMind-AI/

│

├── ai-engine/

│   ├── app/

│   │   ├── main.py

│   │   ├── api/

│   │   └── core/

│   ├── .venv/

│   └── requirements.txt

│

├── client/

│   ├── src/

│   │   ├── pages/

│   │   ├── components/

│   │   └── App.jsx

│   ├── package.json

│   └── index.html

│

├── server/

│   ├── src/

│   │   ├── config/

│   │   ├── models/

│   │   ├── controllers/

│   │   ├── routes/

│   │   └── server.js

│   ├── .env

│   └── package.json

│

├── training/

├── docs/

├── package.json

└── README.md

```

## 11. Prerequisites

Install:

| Software      | Purpose                 |

| ------------- | ----------------------- |

| Git           | Clone repository        |

| Node.js       | Frontend and backend    |

| npm           | JavaScript dependencies |

| Python 3.13   | AI engine               |

| MongoDB Atlas | Cloud database          |

| VS Code       | Development             |

Recommended hardware:

- 8 GB RAM minimum.

- 16 GB RAM recommended.

- Multi-core CPU.

- NVIDIA GPU optional.

- Sufficient storage for dependencies and model files.

## 12. Clone the Repository

Replace the URL with the actual GitHub repository URL.

```

git clone https://github.com/YOUR-USERNAME/DisasterMind-AI.git

cd DisasterMind-AI

```

Install frontend and backend dependencies:

```

npm run install:all

```

## 13. Environment Configuration

### MongoDB Atlas

1. Create a MongoDB Atlas account.

2. Create a cluster.

3. Create a database user.

4. Configure network access.

5. Copy the MongoDB connection URI.

6. Configure the backend environment file.

Create `server/.env`:

```

PORT=5000

CLIENT_URL=http://localhost:5173

AI_ENGINE_URL=http://127.0.0.1:8000

MONGO_URI=mongodb+srv://USERNAME:PASSWORD@YOUR-CLUSTER.mongodb.net/disastermind

JWT_SECRET=YOUR_LONG_RANDOM_SECRET

JWT_EXPIRES_IN=7d

```

Replace example values with your own credentials.

Security:

- Never upload `.env` files to GitHub.

- Never share database credentials.

- Never commit JWT secrets.

- Use strong passwords.

## 14. Python Environment Setup

Open PowerShell:

```

cd D:\DisasterMind-AI\ai-engine

```

Create the environment if it does not already exist:

```

py -3.13 -m venv .venv

```

Activate:

```

.\.venv\Scripts\Activate.ps1

```

If PowerShell blocks activation:

```

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

```

Then activate:

```

.\.venv\Scripts\Activate.ps1

```

Install dependencies:

```

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```

Do not recreate an existing environment unnecessarily.

## 15. GPU and CPU Setup

### NVIDIA GPU Users

Check GPU:

```

nvidia-smi

```

Check PyTorch:

```

python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"

```

### Users Without a GPU

A dedicated GPU is not mandatory.

Activate the AI-engine virtual environment and install CPU-compatible PyTorch:

```

cd D:\DisasterMind-AI\ai-engine

.\.venv\Scripts\Activate.ps1

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

```

Verify:

```

python -c "import torch; print('CUDA available:',torch.cuda.is_available()); print('CPU execution is available')"

```

To hide CUDA devices in the current PowerShell session:

```

$env:CUDA_VISIBLE_DEVICES=""

```

CPU users can run the application without a dedicated GPU. Deep Learning operations may take longer.

GPU acceleration is optional, not mandatory.

## 16. How to Start the Complete Project

Three terminals are required for normal local development.

MongoDB Atlas runs in the cloud, so it does not need a separate local terminal.

### TERMINAL 1 — Python AI Engine

```

cd D:\DisasterMind-AI\ai-engine

.\.venv\Scripts\Activate.ps1

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```

AI Engine:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

Swagger documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health:

[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

GPU:

[http://127.0.0.1:8000/gpu](http://127.0.0.1:8000/gpu)

Keep this terminal running.

### TERMINAL 2 — Node.js Backend

Open a second PowerShell terminal:

```

cd D:\DisasterMind-AI

npm run server

```

Backend:

[http://localhost:5000](http://localhost:5000)

Health:

[http://localhost:5000/api/health](http://localhost:5000/api/health)

Keep this terminal running.

### TERMINAL 3 — React Frontend

Open a third PowerShell terminal:

```

cd D:\DisasterMind-AI

npm run client

```

Frontend:

[http://localhost:5173](http://localhost:5173)

Open this URL in your browser.

Keep this terminal running.

## 17. Main Commands Reference

### Root Project

```

cd D:\DisasterMind-AI

npm run install:all

npm run server

npm run client

```

### AI Engine

```

cd D:\DisasterMind-AI\ai-engine

.\.venv\Scripts\Activate.ps1

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```

Development mode with automatic reload:

```

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

Install Python requirements:

```

python -m pip install -r requirements.txt

```

Check installed packages:

```

python -m pip list

```

### Frontend

```

cd D:\DisasterMind-AI\client

npm install

npm run dev

```

Build frontend:

```

npm run build

```

### Backend

```

cd D:\DisasterMind-AI\server

npm install

npm run dev

```

## 18. Main API Endpoints

### Python AI Engine

| Method | Endpoint                         | Purpose                              |

| ------ | -------------------------------- | ------------------------------------ |

| GET    | `/`                              | Service information                  |

| GET    | `/health`                        | Health check                         |

| GET    | `/gpu`                           | GPU status                           |

| POST   | `/api/v1/predictions/earthquake` | Earthquake occurrence classification |

| POST   | `/api/v1/generator/counts`       | Synthetic count generation           |

| POST   | `/api/v1/scenarios/generate`     | Scenario generation                  |

Swagger documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Node.js Backend

| Method  | Endpoint             | Purpose           |

| ------- | -------------------- | ----------------- |

| POST    | `/api/auth/register` | User registration |

| POST    | `/api/auth/login`    | User login        |

| GET     | `/api/health`        | Backend health    |

| Various | `/api/ai`            | AI gateway routes |

## 19. Dependencies and Libraries

### Frontend

| Package          | Purpose               |

| ---------------- | --------------------- |

| react            | UI development        |

| react-dom        | DOM rendering         |

| vite             | Development and build |

| react-router-dom | Navigation            |

| axios            | API communication     |

| leaflet          | Maps                  |

| react-leaflet    | React map integration |

| recharts         | Charts                |

| lucide-react     | Icons                 |

### Backend

| Package      | Purpose                   |

| ------------ | ------------------------- |

| express      | REST API                  |

| mongoose     | MongoDB modelling         |

| bcryptjs     | Password hashing          |

| jsonwebtoken | Authentication            |

| dotenv       | Environment configuration |

| cors         | Cross-origin requests     |

### Python

| Library      | Purpose                             |

| ------------ | ----------------------------------- |

| fastapi      | AI API framework                    |

| uvicorn      | ASGI server                         |

| numpy        | Numerical computation               |

| pandas       | Data processing                     |

| scipy        | Scientific computation              |

| scikit-learn | Machine Learning                    |

| torch        | Deep Learning                       |

| joblib       | Model serialization                 |

| pydantic     | Data validation                     |

| matplotlib   | Visualization                       |

| plotly       | Interactive visualization           |

| pgmpy        | Probabilistic modelling experiments |

Exact installed versions are defined by the project's dependency files.

## 20. Important Developer Instructions

1. Start the AI engine first.

2. Start the Node.js backend second.

3. Start the React frontend third.

4. Keep all three terminals open.

5. Ensure MongoDB Atlas is accessible.

6. Check `MONGO_URI` in `server/.env`.

7. Never upload secrets or `.env` files.

8. Do not delete trained model files required by the AI engine.

9. Keep training and inference preprocessing consistent.

10. Use chronological evaluation for time-dependent data.

11. Do not interpret anomaly scores as earthquake probabilities.

12. Do not interpret synthetic scenarios as actual future events.

13. Do not claim exact earthquake prediction.

14. Keep Python dependencies inside the virtual environment.

15. GPU is optional; CPU execution is supported with compatible dependencies.

### Troubleshooting

Check Python interpreter:

```

python -c "import sys; print(sys.executable)"

```

It should point to:

```

D:\DisasterMind-AI\ai-engine\.venv\Scripts\python.exe

```

Check core Python imports:

```

python -c "import fastapi, uvicorn, numpy, pandas, sklearn, torch, joblib; print('Core imports successful')"

```

If the backend cannot connect to MongoDB:

- Check the `.env` file.

- Check the MongoDB URI.

- Check database credentials.

- Check Atlas network access.

If the frontend cannot connect:

- Ensure port 5000 is running.

- Ensure port 5173 is running.

- Check API URLs and browser console errors.

## 21. Future Improvements

Potential future extensions:

- Flood prediction and analysis.

- Cyclone analysis.

- Wildfire analysis.

- Landslide analysis.

- Improved geographical scenario generation.

- Better modelling of extreme earthquake-count patterns.

- Model calibration and uncertainty analysis.

- Explainable AI.

- Role-based access control.

- Exportable reports.

- Model monitoring.

- Cloud deployment.

These are potential improvements, not claims about currently implemented features.

## 22. Project Impact

DisasterMind AI demonstrates the integration of:

- Full-stack web development.

- REST API architecture.

- Cloud database integration.

- Data engineering.

- Machine Learning.

- Deep Learning.

- Statistical scenario generation.

- Geospatial visualization.

- Model evaluation.

It provides a practical example of applying AI and software engineering to a real-world disaster-analysis problem while acknowledging the uncertainty and limitations of earthquake modelling.

## 23. Data Sources and Documentation

### USGS Earthquake Data

[https://earthquake.usgs.gov/fdsnws/event/1/](https://earthquake.usgs.gov/fdsnws/event/1/)

### Technology Documentation

- React: [https://react.dev/](https://react.dev/)

- Vite: [https://vite.dev/](https://vite.dev/)

- Node.js: [https://nodejs.org/](https://nodejs.org/)

- Express: [https://expressjs.com/](https://expressjs.com/)

- MongoDB: [https://www.mongodb.com/docs/](https://www.mongodb.com/docs/)

- FastAPI: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

- Python: [https://www.python.org/](https://www.python.org/)

- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

- PyTorch: [https://pytorch.org/](https://pytorch.org/)

## 24. License

Add an appropriate open-source license before publishing this repository.

Without a license, users should not assume permission to reuse, modify, or redistribute the project.

DisasterMind AI — Turning Historical Disaster Data into Analytical Intelligence.
