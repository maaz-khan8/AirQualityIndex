# Air Quality Index Forecasting System

A comprehensive machine learning system for forecasting Air Quality Index (AQI) using historical air quality and weather data. The system implements multi-horizon forecasting (1h, 6h, 12h, 24h, 48h, 72h ahead) using ensemble machine learning models, with all data stored and managed in Hopsworks cloud infrastructure.


## Overview

This project provides an end-to-end machine learning pipeline for air quality forecasting that:

- Fetches historical air quality and weather data from Open-Meteo API
- Engineers comprehensive features including time-based, lag, rolling window, and derived features
- Calculates EPA-compliant Air Quality Index (AQI) from pollutant concentrations
- Trains multi-horizon forecasting models (Random Forest and Ridge Regression)
- Stores all data, features, and models exclusively in Hopsworks cloud storage
- Provides interactive dashboard for visualization and real-time predictions
- Implements automated alert system for air quality thresholds
- Performs model interpretability analysis using SHAP

## Features

### Data Management
- Cloud-first architecture: All data stored in Hopsworks Feature Store (no local storage)
- Historical data fetching from Open-Meteo API
- Automated feature engineering pipeline
- EPA-compliant AQI calculation from multiple pollutants

### Machine Learning
- Multi-horizon forecasting (1h, 6h, 12h, 24h, 48h, 72h ahead)
- Ensemble models: Random Forest and Ridge Regression
- Multi-output regression for simultaneous horizon predictions
- Model versioning and metrics tracking in Hopsworks Model Registry
- Automatic model retraining capability

### Visualization & Monitoring
- Interactive Streamlit dashboard with real-time AQI display
- Multi-horizon forecast visualization
- Model performance metrics and comparisons
- Exploratory Data Analysis (EDA) snapshots
- Alert system with severity levels

### Model Interpretability
- SHAP-based feature importance analysis
- Model explainability visualizations
- Feature contribution analysis

## Architecture

The system follows a cloud-centric architecture where:

1. **Data Layer**: Open-Meteo API → Hopsworks Feature Store
2. **Feature Engineering**: Raw data → Engineered features → Hopsworks Feature Group
3. **Training Layer**: Hopsworks Feature Store → ML Models → Hopsworks Model Registry
4. **Inference Layer**: Hopsworks Models → Dashboard → Predictions
5. **Monitoring**: Alert System + EDA + SHAP Analysis

All intermediate data storage is handled by Hopsworks, ensuring no local data persistence.

## Prerequisites

- Python 3.8 or higher
- Hopsworks account with API key
- Internet connection for API data fetching
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AirQualityIndex
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv312
venv312\Scripts\activate

# Linux/Mac
python -m venv venv312
source venv312/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
```

Alternatively, set the environment variable directly:

```bash
# Windows PowerShell
$env:HOPSWORKS_API_KEY="your_api_key"

# Linux/Mac
export HOPSWORKS_API_KEY="your_api_key"
```

## Configuration

The `config.py` file contains all configuration parameters:

### API Configuration
- `HOPSWORKS_API_KEY`: Your Hopsworks API key (required)
- `HOPSWORKS_PROJECT_NAME`: Your Hopsworks project name
- `HOPSWORKS_FEATURE_GROUP_NAME`: Feature group name in Hopsworks
- `HOPSWORKS_FEATURE_GROUP_VERSION`: Feature group version
- `HOPSWORKS_MODEL_NAME`: Model registry name

### Location Settings
- `LOCATION`: Dictionary with city, latitude, longitude, and timezone
- Default: London, UK (51.5074, -0.1278)

### Feature Engineering
- `LAG_HOURS`: List of lag hours for temporal features (default: [1, 6])
- `ROLLING_WINDOWS`: Rolling window sizes in hours (default: [12, 24, 48])
- `FORECAST_HORIZONS`: Prediction horizons in hours (default: [1, 6, 12, 24, 48, 72])

### Model Configuration
- `MODELS_CONFIG`: Hyperparameters for Random Forest and Ridge Regression models
- `TRAIN_TEST_SPLIT`: Training/test split ratio (default: 0.8)

### Data Features
- `POLLUTANT_FEATURES`: List of air quality pollutants to fetch
- `WEATHER_FEATURES`: List of weather parameters to fetch

## Usage

### Command Line Interface

The system provides a unified CLI interface through `src/main.py`:

```bash
python -m src.main <command> [options]
```

### Available Commands

#### Setup (Initial Data Collection and Training)

Fetches historical data (default: 1 year), engineers features, uploads to Hopsworks, and trains models:

```bash
python -m src.main setup
```

With custom date range:

```bash
python -m src.main setup --start 2024-01-01 --end 2024-12-31
```

#### Daily Update

Fetches new data from the last 24 hours, appends to Hopsworks, and retrains models:

```bash
python -m src.main update
```

#### Dashboard

Launches the interactive Streamlit dashboard:

```bash
python -m src.main dashboard
```

The dashboard will be available at `http://localhost:8501`

#### Test Connections

Tests Hopsworks connection and data fetcher:

```bash
python -m src.main test
```

### Workflow Examples

#### Initial Setup

1. Configure your `.env` file with Hopsworks credentials
2. Run initial setup:
   ```bash
   python -m src.main setup
   ```
3. This will:
   - Fetch 1 year of historical data
   - Generate EDA snapshots
   - Engineer features
   - Upload to Hopsworks Feature Store
   - Train multi-horizon models
   - Save models to Hopsworks Model Registry
   - Run alert system and interpretability analysis

#### Daily Operations

1. Run daily update:
   ```bash
   python -m src.main update
   ```
2. This will:
   - Fetch last 24 hours of data
   - Append to existing feature group
   - Retrain models with updated data
   - Update model registry

#### Viewing Results

1. Launch dashboard:
   ```bash
   python -m src.main dashboard
   ```
2. Access visualizations, forecasts, and metrics

## Project Structure

```
AirQualityIndex/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # This file
│
├── src/                      # Source code directory
│   ├── main.py              # Main entry point and CLI
│   ├── pipeline.py          # Unified pipeline orchestrator
│   ├── data_fetcher.py      # Open-Meteo API data fetching
│   ├── feature_engineering.py  # Feature creation and engineering
│   ├── aqi_calculator.py    # EPA-compliant AQI calculation
│   ├── training.py          # Model training (MultiHorizonForecaster)
│   ├── hopsworks_client.py  # Hopsworks integration
│   ├── dashboard.py         # Streamlit dashboard
│   ├── alerts.py            # AQI alert system
│   ├── eda.py               # Exploratory Data Analysis
│   ├── interpretability.py  # SHAP model interpretability
│   └── metrics_loader.py   # Model metrics loading
│
├── logs/                     # Application logs
│   └── aqi_project.log     # Main log file
│
├── models/                   # Local model storage (optional, for dashboard)
│   └── *.pkl               # Trained model files
│
├── reports/                  # EDA reports and visualizations
│   └── latest/              # Latest EDA snapshots
│
├── artifacts/                # SHAP analysis artifacts
│   └── *.json, *.png        # Interpretability results
│
└── alerts/                   # Alert history
    └── alert_history.json   # Alert records
```

## Components

### Data Fetcher (`data_fetcher.py`)

The `OpenMeteoFetcher` class handles API interactions:

- `fetch_air_quality_data()`: Fetches air quality data from Open-Meteo
- `fetch_weather_data()`: Fetches weather data from Open-Meteo historical API
- `fetch_combined_historical_data()`: Combines air quality and weather data

### Feature Engineering (`feature_engineering.py`)

The `FeatureEngineer` class creates comprehensive features:

- **Time Features**: Hour, day of week, month, season, cyclic encoding
- **Derived Features**: Temperature-humidity interaction, wind chill, PM ratios
- **Lag Features**: Historical values at multiple time lags
- **Rolling Features**: Mean and standard deviation over rolling windows
- **Target Variable**: AQI values shifted forward for forecasting

### AQI Calculator (`aqi_calculator.py`)

EPA-compliant AQI calculation:

- Supports PM2.5, PM10, O3, NO2, SO2, CO
- Unit conversions (Open-Meteo units to EPA standards)
- AQI category classification (Good, Moderate, Unhealthy, etc.)
- Maximum AQI calculation across all pollutants

### Training (`training.py`)

Multi-horizon forecasting models:

- **MultiHorizonForecaster**: Multi-output regression for all horizons
- **AQIForecaster**: Single-horizon forecasting (legacy, for reference)
- Models: Random Forest and Ridge Regression
- Metrics: MSE, MAE, R² for each horizon

### Hopsworks Client (`hopsworks_client.py`)

Cloud storage integration:

- `connect()`: Establish Hopsworks connection
- `create_feature_group()`: Upload engineered features
- `get_training_data()`: Retrieve data for training
- `get_feature_data()`: Retrieve data for dashboard
- `save_model()`: Save models to Model Registry
- `load_models()`: Load models for inference

### Pipeline (`pipeline.py`)

Unified pipeline orchestrator:

- **Initial Setup**: Full data collection and model training
- **Daily Update**: Incremental data and retraining
- **Alert System**: Automated AQI threshold monitoring
- **Interpretability**: SHAP analysis integration

### Dashboard (`dashboard.py`)

Streamlit-based interactive interface:

- Current AQI display with gauge visualization
- Multi-horizon forecast predictions (1h to 72h)
- Time series charts for pollutants and weather
- Model performance metrics
- Alert panel
- EDA snapshot viewer

### Alert System (`alerts.py`)

Automated air quality monitoring:

- Severity levels: Low, Moderate, High, Critical
- Threshold-based alerting
- Duplicate prevention
- Alert history tracking
- Recommendations generation

## Workflow

### Initial Setup Workflow

1. **Data Fetching**: Retrieve historical air quality and weather data from Open-Meteo API
2. **EDA Generation**: Create exploratory data analysis snapshots
3. **Feature Engineering**: Calculate AQI, create time/lag/rolling features
4. **Hopsworks Upload**: Store engineered features in Feature Store
5. **Model Training**: Train multi-horizon models on Hopsworks data
6. **Model Saving**: Upload trained models to Hopsworks Model Registry
7. **Alert System**: Run initial alert evaluation
8. **Interpretability**: Generate SHAP analysis for model explainability

### Daily Update Workflow

1. **Incremental Data**: Fetch last 24 hours of data
2. **Feature Engineering**: Apply same feature engineering pipeline
3. **Data Append**: Add new features to existing Hopsworks Feature Group
4. **Model Retraining**: Retrain models with updated dataset
5. **Model Update**: Save new model versions to Registry
6. **Alert Evaluation**: Check for new alerts based on updated data

### Dashboard Workflow

1. **Data Loading**: Fetch latest data from Hopsworks Feature Store
2. **Model Loading**: Load trained models from Model Registry or local cache
3. **Feature Preparation**: Prepare latest data point for prediction
4. **Forecast Generation**: Generate multi-horizon predictions
5. **Visualization**: Display charts, metrics, and alerts

## API Documentation

### Open-Meteo API

The system uses two Open-Meteo endpoints:

- **Air Quality API**: `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Historical Weather API**: `https://archive-api.open-meteo.com/v1/archive`

Both APIs require:
- `latitude` and `longitude` parameters
- `start_date` and `end_date` in YYYY-MM-DD format
- `hourly` parameter with comma-separated feature names

### Hopsworks API

The system integrates with Hopsworks for:
- Feature Store: Data storage and retrieval
- Model Registry: Model versioning and metrics
- Feature Groups: Versioned feature storage
- Feature Views: Query interface for features
