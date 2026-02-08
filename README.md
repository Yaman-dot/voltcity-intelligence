# VoltCity Charge: AI-Driven EV Network Optimization

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

VoltCity Charge is an end-to-end framework for managing Electric Vehicle (EV) charging infrastructure. The project integrates predictive analytics, an ethical AI pipeline for bias mitigation, and a Reinforcement Learning agent to optimize user charging behavior.

## Key Features

* **Multi-Model Intelligence:** Combines regression for cost estimation and classification for session type analysis.
* **Algorithmic Fairness:** Implements a three-stage mitigation pipeline (Pre-, In-, and Post-processing) to eliminate bias in features like `User Type` and `Battery Capacity`.
* **Reinforcement Learning:** Features a Q-Learning agent with 120 discrete states designed to recommend optimal charging actions (Delay, Move, or Proceed) based on immediate rewards.
* **XAI Integration:** Utilizes LIME and SHAP handlers to provide human-readable explanations for AI-driven recommendations.
* **Interactive Web Dashboard:** A Flask-based interface featuring an admin panel and car dashboard for real-time monitoring.

## Technical Architecture

### 1. Fairness & Mitigation Pipeline
The project audits models for "Potential Bias" using metrics like Demographic Parity (DP) and Disparate Impact Ratio (DIR).
* **Preprocessing:** SMOTE for data balancing.
* **In-Processing:** Sample weighting during model training.
* **Post-Processing:** Threshold adjustment to ensure "Fair" status for protected groups.

### 2. Reinforcement Learning Environment
The agent manages session optimization through a discrete state-space model:
* **State Space:** 120 states (Battery Level, Location, Charger Type, Time of Day).
* **Reward Function:** Optimized to minimize charging costs ($R=2-(3 \times \text{Charging Cost})$).
* **Strategy:** Hybrid Master-Slave architecture combining global grid constraints with personalized user optimization.

## Project Structure

```bash
├── data/               # Processed datasets
├── explainers/         # XAIHandlers (LIME/SHAP)
├── notebooks/          # Preprocessing, Modelling, Bias and Fairness Mitigations
├── saved_models/       # Serialized models and RL Q-tables
├── static/             # CSS and frontend assets for the dashboard
├── templates/          # HTML templates for admin and car interfaces
├── app.py              # Main Flask application entry point
└── requirements.txt    # Project dependencies
```

## Tech Stack
* **Languages**: Python, C++
* **AI/ML**: XGBoost, Scikit-learn, Q-Learning
* **Ethical AI**: AI Fairness 360 (AIF360) concepts
* **Web**: Flask, HTML5, CSS3

## Getting Started
Clone the repository:
```bash
git clone [https://github.com/Yaman-dot/VoltCity-Charge.git](https://github.com/Yaman-dot/VoltCity-Charge.git)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the Flask App:
```bash
python app.py
```
