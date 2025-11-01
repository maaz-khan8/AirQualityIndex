import os
from datetime import datetime, timedelta

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(config_dir, '.env')
    
    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        if os.path.exists(env_path):
            result = load_dotenv(dotenv_path=env_path, override=True)
            if result:
                return
        # Fallback to current directory
        load_dotenv(dotenv_path='.env', override=True)
        return
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: manually parse .env file
    env_files = [env_path, '.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            # Set environment variable (override=True for .env file)
                            if key:
                                os.environ[key] = value
                return
            except Exception:
                continue

load_env_file()

# API Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_maazkhan")

# Validate required environment variables
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is required but not set")
HOPSWORKS_FEATURE_GROUP_NAME = "london_air_quality_6h"
HOPSWORKS_FEATURE_GROUP_VERSION = 5
HOPSWORKS_FEATURE_VIEW_NAME = "london_air_quality_6h_view"
HOPSWORKS_FEATURE_VIEW_VERSION = 5
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
LAG_HOURS = [1, 6]
ROLLING_WINDOWS = [12, 24, 48]
FORECAST_HORIZON = 6

FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]  

# Model Configuration (Based on Key_Features.md)
MODELS_CONFIG = {
    'random_forest': {
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'ridge_regression': {
        'alpha': 20.0,
        'random_state': 42
    }
}

# Training
TRAIN_TEST_SPLIT = 0.8
METRICS = ['mse', 'rmse', 'mae', 'r2', 'mape']

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(message)s"
LOG_FILE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOGS_DIR = "logs"
LOG_FILE = f"{LOGS_DIR}/aqi_project.log"
PLOTS_DIR = "plots"

os.makedirs(LOGS_DIR, exist_ok=True)