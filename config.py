import os
from datetime import datetime, timedelta

# API Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_maazkhan")

# Validate required environment variables
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is required but not set")
HOPSWORKS_FEATURE_GROUP_NAME = "london_air_quality_6h"
HOPSWORKS_FEATURE_GROUP_VERSION = 1
HOPSWORKS_FEATURE_VIEW_NAME = "london_air_quality_6h_view"
HOPSWORKS_FEATURE_VIEW_VERSION = 1
HOPSWORKS_MODEL_NAME = "aqi_6h_forecast"

# Open-Meteo API
OPENMETEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Location
LOCATION = {
    "city": "London",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "timezone": "Europe/London"
}

# Data Collection
HISTORICAL_START_DATE = "2024-10-16"
HISTORICAL_END_DATE = "2025-10-16"

# Features
POLLUTANT_FEATURES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
    "sulphur_dioxide", "ozone", "dust", "uv_index"
]

WEATHER_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl", 
    "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover"
]

# Feature Engineering
LAG_HOURS = [1, 2, 3, 6]
ROLLING_WINDOWS = [3, 6, 12, 24]
FORECAST_HORIZON = 6

# Multi-Horizon Forecasting (3-day ahead predictions)
FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]  # 1h, 6h, 12h, 24h, 48h, 72h (3 days)

# Model Configuration (Based on Key_Features.md)
MODELS_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'ridge_regression': {
        'alpha': 1.0,
        'random_state': 42
    }
}

# Training
TRAIN_TEST_SPLIT = 0.8
METRICS = ['mse', 'rmse', 'mae', 'r2', 'mape']

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOGS_DIR = "logs"
LOG_FILE = f"{LOGS_DIR}/aqi_project.log"
PLOTS_DIR = "plots"

# Ensure logs directory exists
import os
os.makedirs(LOGS_DIR, exist_ok=True)