# 🌊 AquaVision: AI-Powered Automated Underwater Marine Debris & Anomaly Detection System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg?style=flat&logo=react)](https://react.dev/)
[![YOLO](https://img.shields.io/badge/AI%20Model-Physics--Constrained%20YOLO-blue.svg)](https://docs.ultralytics.com/)
[![TailwindCSS](https://img.shields.io/badge/UI-TailwindCSS%203.4-38B2AC.svg?style=flat&logo=tailwind-css)](https://tailwindcss.com/)
[![Vite](https://img.shields.io/badge/Bundler-Vite%207-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev/)

**AquaVision** is an end-to-end decision-support platform designed for autonomous underwater vehicles (AUVs), marine survey teams, and ocean conservation organizations. Built for the **Smart India Hackathon (SIH)**, AquaVision processes raw, low-contrast side-scan sonar (SSS) imagery, suppresses acoustic speckle noise and swath illumination decay, accurately detects underwater anomalies across 4 critical classes, calculates environmental and navigational risk scores, plans optimal recovery routes, and visualizes operations through an interactive geospatial command dashboard.

---

## 📌 Key Capabilities

- **Physics-Constrained Preprocessing**: A 4-stage deterministic image transformation pipeline (Cross-Track Swath Illumination Normalization, Bilateral Edge-Preserving Denoising, Robust 1%–99% Percentile Normalization, and CIELAB Contrast-Limited Adaptive Histogram Equalization - CLAHE).
- **Calibrated Multi-Class Sonar Detector**: Physics-tuned confidence thresholds tailored specifically to the acoustic shadow and highlight signatures of 4 target classes:
  - 🚢 **Shipwreck** (`0.25` threshold): High precision on macro-structures, suppressing diffuse seafloor reverberation.
  - ✈️ **Aircraft** (`0.21` threshold): Captures aerodynamic geometries and distinct shadow tails.
  - 💣 **Mine / UXO** (`0.10` threshold): High sensitivity recall on hazardous micro-targets (<20 px).
  - 🕸️ **Fishing Gear / Ghost Nets** (`0.16` threshold): Resolves diffuse, low-contrast acoustic reflections from abandoned nets and traps.
- **Dynamic Decision Support & Risk Engine**: Multi-factor risk scoring (0–100) based on object class hazard, bounding-box confidence, bathymetric depth, and estimated target size.
- **Priority & GIS Mission Planner**: Intelligent clustering and route optimization using a distance-first, priority-tie-breaking algorithm to generate fuel-efficient, safety-prioritized survey and cleanup routes.
- **Full-Stack Mission Control Dashboard**: Real-time sonar viewer, geospatial mapping (Leaflet), verification workflows, before/after cleanup tracking, and telemetry reports.

---

## 🗂️ Clean Repository Architecture

```text
AquaVision/
├── frontend/                 # Interactive React + Vite + TailwindCSS Mission Control UI
│   ├── src/                  # Components, Pages, State Context, and API Clients
│   ├── public/               # Static assets, icons, and benchmark visualizations
│   └── package.json          # Node dependencies & Vite build configuration
│
├── backend/                  # Production FastAPI Application & Database Layer
│   ├── app/                  # Main server, ORM models, schemas, and engines
│   │   ├── main.py           # REST API endpoints & detection pipeline orchestration
│   │   ├── models.py         # SQLAlchemy database models
│   │   ├── schemas.py        # Pydantic data schemas
│   │   ├── model/            # Production model weights (aqua_vision_100ep_best.pt)
│   │   └── preprocessing/    # Self-contained sonar image transformation pipeline
│   ├── requirements.txt      # Python dependencies for backend runtime
│   └── start_backend.bat     # Windows startup script
│
├── model/                    # Production Model Hub
│   ├── aqua_vision_100ep_best.pt  # 100-Epoch physics-constrained trained weights
│   ├── evaluate_sonar_model.py    # Batch evaluation script (PR/F1 curves, confusion matrix)
│   ├── run_sonar_inference_demo.py# Standalone CLI inference demo with bounding box overlays
│   └── README.md             # Model technical specifications, classes & metrics
│
├── gis/                      # Geospatial Routing & Mission Planning Subsystem
│   ├── main.py               # Mission plan generator & CLI runner
│   ├── geo_service.py        # Haversine distance, coordinate validator & data normalizer
│   ├── route_planner.py      # Spatial clustering & priority-based route optimizer
│   ├── priority_engine.py    # Standalone priority scoring engine
│   ├── detections.json       # Mock sample detection coordinates for local validation
│   └── tests/                # Automated unit test suite (test_member5.py)
│
├── computer_vision/          # Core Sonar Signal Processing Algorithms
│   ├── final_preprocessing_pipeline.py # 4-Stage physics-informed preprocessing
│   ├── final_preprocessing_config.json # Calibrated spatial & frequency filter parameters
│   ├── clahe.py              # Contrast-Limited Adaptive Histogram Equalization
│   ├── denoising.py          # Bilateral edge-preserving filtering
│   ├── speckle_filter.py     # Classical Lee & Frost speckle suppression
│   ├── swath_normalization.py# Range-dependent cross-track attenuation compensation
│   └── README.md             # Preprocessing theory & pipeline documentation
│
│
├── start_backend.bat         # Single-click launcher for the FastAPI backend server
├── .gitignore                # Comprehensive exclusion rules for GitHub cleanliness
└── README.md                 # Root project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: Version `3.10` or `3.11` recommended.
- **Node.js**: Version `18.x` or `20.x` with `npm`.
- **Git**: For repository version control.

---

### 2. Backend Setup & Startup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv311
   # On Windows (PowerShell):
   .\venv311\Scripts\Activate.ps1
   # On Windows (Command Prompt):
   .\venv311\Scripts\activate.bat
   # On Linux / macOS:
   source venv311/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment settings:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *(By default, SQLite is automatically used if PostgreSQL is not configured).*

5. Start the backend server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Alternatively on Windows, simply double-click `start_backend.bat` in the root folder.*

   API documentation will be accessible at:
   - Interactive Swagger Docs: **http://127.0.0.1:8000/docs**
   - ReDoc: **http://127.0.0.1:8000/redoc**

---

### 3. Frontend Setup & Startup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Launch the Vite development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   - Application URL: **http://localhost:5173**

---

### 4. GIS & Route Planning Verification

The GIS mission planner can be run independently to compute waypoint plans from detection coordinates:

```bash
cd gis
python main.py
```

To run the automated GIS unit tests:
```bash
python tests/test_member5.py
```

---

## 🧠 Model Usage & Evaluation

The production model weights are stored in `model/aqua_vision_100ep_best.pt`.

### Run Standalone Inference Demo
To process a side-scan sonar image and save annotated visual detections:
```bash
python model/run_sonar_inference_demo.py \
    --source path/to/sonar_image.jpg \
    --weights model/aqua_vision_100ep_best.pt \
    --output inference_outputs/
```

### Run Model Evaluation
To evaluate precision, recall, mAP@50, and class-specific metrics:
```bash
python model/evaluate_sonar_model.py \
    --dataset path/to/dataset \
    --weights model/aqua_vision_100ep_best.pt
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/predict` | `POST` | Upload single sonar image for 4-stage preprocessing and YOLO inference |
| `/surveys` | `GET` / `POST` | Create or list survey missions with geospatial metadata |
| `/detections` | `GET` / `PATCH`| Query detections filtered by risk/priority; update cleanup status |
| `/route` | `POST` | Compute optimal vessel route based on active detections |
| `/verify` | `POST` | Submit before/after sonar imagery to verify target recovery |
| `/stats` | `GET` | Retrieve global mission metrics, risk breakdowns, and completion rates |

---

## 📚 Documentation & Research Reports

Detailed documentation is organized under [`docs/`](docs/):
- **Architecture Blueprints**: [`docs/architecture/`](docs/architecture/)
- **100-Epoch Benchmark Analysis**: [`docs/benchmarks/`](docs/benchmarks/)
- **Ablation Studies & Diagnostics**: [`docs/reports/`](docs/reports/)
- **Subsystem Integration Details**: [`docs/integration/`](docs/integration/)

---

## 🔒 License & Intellectual Property

Developed for the **Smart India Hackathon (SIH)**. All rights reserved by the AquaVision Project Team.
