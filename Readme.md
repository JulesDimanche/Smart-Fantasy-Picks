# Smart Fantasy Picks – Dual-Sport Fantasy Team Prediction Platform

## Overview
Smart Fantasy Picks is a modular fantasy sports prediction platform that uses Machine Learning to generate optimal fantasy teams for **Cricket** and **Football**. The system is built using Flask Blueprints, allowing each sport to function as an independent subsystem with its own models, data pipelines, and UI.

---

## System Purpose

### Cricket Prediction Engine
- XGBoost regression models
- Multi-factor scoring combining:
  - ML predictions  
  - Venue performance  
  - Recent form  
  - Head‑to‑head matchups  

### Football Prediction Engine
- TensorFlow/Keras neural network  
- Predicts fantasy points  
- Applies formation + position constraints to build a valid XI  

Each module runs independently with no shared logic.

---

## Application Entry Points

| Component | Path / File | Purpose |
|----------|--------------|---------|
| Landing Page | `index.html` | Sport selection interface |
| Cricket Module | `/cricket` | XGBoost‑based team prediction |
| Football Module | `/football` | Neural network‑based team prediction |

---

## System Architecture Overview

*Add workflow / architecture image here:*  
`![Architecture Diagram](path/to/your_image.png)`

Architecture follows:
- `main_app.py` as core Flask container  
- Independent Blueprint modules (`cricket_bp`, `football_bp`)  
- Separate templates, ML models, and data processors  

---

## Request Flow Pattern
User requests flow as follows:
- `@app.route('/')` → loads the landing page  
- `/cricket/*` → forwarded to `cricket_bp`  
- `/football/*` → forwarded to `football_bp`  
- Each blueprint handles its own templates, inputs, and predictions  

Cricket uses the **FantasyTeamSelector** class,  
Football uses a **select_fantasy_team()** function.

---

## Core Components

### Flask Application Core (`main_app.py`)
- Initializes Flask app  
- Registers Blueprints  
- Serves landing page  
- Minimal logic (all heavy lifting done in modules)

---

## Blueprint Architecture

| Blueprint | Import Path | URL Prefix | Location |
|----------|-------------|------------|----------|
| cricket_bp | `cricket.app` | `/cricket` | `cricket/app.py` |
| football_bp | `football.football` | `/football` | `football/football.py` |

---

## Machine Learning Models

### Cricket Models
- `batsman_model.joblib`
- `bowler_model.joblib`
- `allrounder_model.joblib`
- Corresponding scalers (`*_scaler.joblib`)

### Football Model
- `fantasy_points_model.h5` (Keras/TensorFlow)

---

## Data Flow Architecture

**Cricket**
- Client uploads CSV  
- Server processes data  
- Multiple scoring stages  

**Football**
- Text-based input  
- Single neural network prediction  
- Constraint solver for valid XI  

---

## Module Independence

| Aspect | Cricket | Football |
|--------|---------|----------|
| ML Framework | XGBoost, sklearn | TensorFlow/Keras |
| Model Count | 6 files | 1 file |
| UI | Rich interactive gallery | Simple form input |
| Scoring | Multi-factor | Neural network |
| Data Source | Multiple CSVs | Text input |

Independent development with no shared prediction logic.

---

## Landing Page Portal

The `index.html` landing page includes:
- particles.js animation  
- Animated selection cards  
- Simple routing via JavaScript  

---

## Technology Stack

**Backend**
- Flask, Jinja2  

**Frontend**
- HTML5, CSS3, JS  
- Particles.js  
- Animate.css  

**ML**
- XGBoost, scikit‑learn  
- TensorFlow/Keras  
- Pandas, NumPy  
- Joblib, HDF5  

---

## Key Design Decisions
- Blueprint‑based modular architecture  
- Different ML frameworks per sport  
- Independent UI and backend flows  
- Pre-trained offline models loaded at runtime  
- No shared prediction logic  

---

## Add Workflow / Architecture Diagram Here
```
![Workflow Diagram](path/to/diagram.png)
```

---

## License
Add your project license here.

