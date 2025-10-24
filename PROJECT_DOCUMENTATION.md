# Air Quality Index - 6-Hour Ahead Forecasting Project

## 🎯 Project Overview

This project implements a complete machine learning pipeline for **6-hour ahead AQI (Air Quality Index) forecasting** using London air quality data. The system combines real-time data collection, feature engineering, cloud storage, and model training to predict air quality conditions.

## ✅ What We Achieved

### 📊 **Data Pipeline**
- ✅ Collected **9,192 hourly records** from Open-Meteo API (1 year of London data)
- ✅ Integrated air quality + weather data from free APIs
- ✅ Created **138 engineered features** for time-series forecasting
- ✅ Uploaded data to Hopsworks Feature Store (cloud storage)

### 🤖 **Machine Learning**
- ✅ Trained **XGBoost** and **Random Forest** models
- ✅ **Random Forest** performs best: R² = 0.405, MAE = 7.01
- ✅ Models saved to Hopsworks Model Registry
- ✅ Feature importance analysis completed

### 🌐 **Dashboard & Visualization**
- ✅ Interactive Streamlit dashboard
- ✅ Real-time AQI monitoring
- ✅ Model performance comparisons
- ✅ Feature importance visualizations

### 🔧 **Technical Stack**
- ✅ **Python 3.12** with virtual environment
- ✅ **Open-Meteo API** (free air quality + weather data)
- ✅ **Hopsworks Feature Store** (cloud ML platform)
- ✅ **XGBoost & Random Forest** (scikit-learn)
- ✅ **Streamlit** (web dashboard)

---

## 📁 Project Structure & File Descriptions

```
AirQualityIndex/
├── config.py                          # Configuration settings
├── requirements.txt                    # Python dependencies
├── PROJECT_DOCUMENTATION.md           # This documentation file
├── logs/                              # Application logs only
│   └── aqi_project.log               # Application logs
├── src/                               # Source code
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # Main pipeline orchestrator
│   ├── data_fetcher.py               # Open-Meteo API client
│   ├── feature_engineering.py        # Feature creation logic
│   ├── hopsworks_client.py           # Hopsworks cloud client
│   ├── training.py                   # Model training logic
│   ├── aqi_calculator.py             # AQI calculation utilities
│   └── dashboard/                    # Web dashboard
│       ├── __init__.py
│       └── app.py                    # Streamlit dashboard
└── venv312/                          # Python 3.12 virtual environment
```

---

## 🔄 Complete Workflow

### **Step 1: Data Collection** 📡
```
Open-Meteo API → London Air Quality + Weather Data → 9,192 hourly records
```
- **File**: `src/data_fetcher.py`
- **Source**: Free Open-Meteo API (no authentication required)
- **Data**: PM2.5, PM10, Ozone, NO₂, SO₂, CO, Temperature, Humidity, Wind, etc.
- **Period**: October 16, 2024 → October 16, 2025 (1 year)

### **Step 2: Feature Engineering** ⚙️
```
Raw Data → Time Features + Lag Features + Rolling Windows → 138 Features
```
- **File**: `src/feature_engineering.py`
- **Features Created**:
  - **Time Features** (12): Hour, day, month, season, cyclical encoding
  - **Lag Features** (40): 1h, 2h, 3h, 6h lags for pollutants/weather
  - **Rolling Features** (56): 3h, 6h, 12h, 24h rolling means/stds
  - **Derived Features** (5): PM ratios, wind chill, interactions
- **Target**: `aqi_6h_ahead` (AQI 6 hours in the future)

### **Step 3: Cloud Storage** ☁️
```
Engineered Features → Hopsworks Feature Store → Cloud Storage
```
- **File**: `src/hopsworks_client.py`
- **Storage**: Hopsworks Feature Store (cloud)
- **Records**: 9,192 samples with 138 features
- **Feature Group**: `london_air_quality_6h v1`
- **Feature View**: `london_air_quality_6h_view`

### **Step 4: Model Training** 🤖
```
Hopsworks Data → XGBoost + Random Forest → Trained Models → Hopsworks Model Registry
```
- **File**: `src/training.py`
- **Models**: XGBoost & Random Forest
- **Training**: 80% train, 20% test split (data retrieved from Hopsworks)
- **Best Model**: Random Forest (R² = 0.405, MAE = 7.01)
- **Storage**: Models and metrics saved to Hopsworks Model Registry only

### **Step 5: Dashboard** 📊
```
Hopsworks Data + Models → Streamlit Dashboard → Interactive Visualizations
```
- **File**: `src/dashboard/app.py`
- **URL**: http://localhost:8501
- **Data Source**: Hopsworks Feature Store and Model Registry
- **Features**: Real-time AQI, model comparisons, feature importance

---

## 📋 File-by-File Breakdown

### **Configuration Files**

#### `config.py`
- **Purpose**: Central configuration for all components
- **Contains**:
  - Hopsworks API credentials
  - Open-Meteo API endpoints
  - Model hyperparameters
  - Feature engineering settings
  - File paths and directories

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Packages**: hopsworks, xgboost, scikit-learn, streamlit, pandas, numpy, plotly

### **Core Pipeline Files**

#### `src/main.py`
- **Purpose**: Main orchestrator for the entire pipeline
- **Commands**:
  - `pipeline`: Run complete workflow
  - `dashboard`: Launch web interface
  - `test`: Test API connections
- **Workflow**: Data collection → Feature engineering → Upload → Training → Dashboard

#### `src/data_fetcher.py`
- **Purpose**: Fetches data from Open-Meteo APIs
- **Class**: `OpenMeteoFetcher`
- **Methods**:
  - `fetch_air_quality_data()`: Gets pollutant data
  - `fetch_weather_data()`: Gets weather data
  - `fetch_combined_historical_data()`: Combines both datasets

#### `src/feature_engineering.py`
- **Purpose**: Creates features for time-series forecasting
- **Class**: `FeatureEngineer`
- **Methods**:
  - `add_time_features()`: Hour, day, month, season features
  - `add_lag_features()`: Historical pollutant/weather values
  - `add_rolling_features()`: Moving averages and standard deviations
  - `add_derived_features()`: PM ratios, wind chill, interactions

#### `src/hopsworks_client.py`
- **Purpose**: Manages all Hopsworks cloud interactions
- **Class**: `HopsworksClient`
- **Methods**:
  - `connect()`: Establishes Hopsworks connection
  - `create_feature_group()`: Uploads data to cloud
  - `create_feature_view()`: Creates queryable data view
  - `get_training_data()`: Retrieves data for training
  - `save_model()`: Saves models to Model Registry

#### `src/training.py`
- **Purpose**: Trains and evaluates ML models
- **Class**: `AQIForecaster`
- **Methods**:
  - `train_from_hopsworks()`: Trains using cloud data
  - `train_from_dataframe()`: Trains using local data
  - `save_models_locally()`: Saves models and metrics
  - `get_feature_importance()`: Analyzes feature importance

#### `src/aqi_calculator.py`
- **Purpose**: Calculates AQI from raw pollutant concentrations
- **Functions**: EPA-compliant AQI calculation algorithms

### **Dashboard Files**

#### `src/dashboard/app.py`
- **Purpose**: Interactive web dashboard
- **Framework**: Streamlit
- **Features**:
  - Real-time AQI gauge
  - Model performance comparisons
  - Feature importance charts
  - Data summary statistics

### **Log Files**

#### `logs/aqi_project.log`
- **Purpose**: Application logging and debugging
- **Content**: Pipeline execution logs, error messages, and status updates

---

## 🚀 How to Run the Project

### **1. Setup Environment**
```bash
# Activate virtual environment
.\venv312\Scripts\activate.ps1

# Install dependencies (already done)
pip install -r requirements.txt
```

### **2. Run Complete Pipeline**
```bash
python -m src.main pipeline
```

### **3. Launch Dashboard**
```bash
python -m src.main dashboard
```
**URL**: http://localhost:8501

### **4. Test Connections**
```bash
python -m src.main test
```

---

## 📊 Model Performance Results

### **Random Forest** (Winner! 🏆)
- **Test R²**: 0.405 (40.5% variance explained)
- **Test MAE**: 7.01 (Mean Absolute Error)
- **Test MAPE**: 20.32% (Mean Absolute Percentage Error)

### **XGBoost**
- **Test R²**: 0.274 (27.4% variance explained)
- **Test MAE**: 7.17 (Mean Absolute Error)
- **Test MAPE**: 20.03% (Mean Absolute Percentage Error)

### **Top 5 Most Important Features**
1. **PM2.5 6h Rolling Mean** (23.9% importance)
2. **PM Combined** (7.1% importance)
3. **AQI 3h Rolling Mean** (5.3% importance)
4. **PM2.5 Current** (4.9% importance)
5. **AQI 6h Rolling Mean** (4.3% importance)

---

## 🌐 Cloud Infrastructure

### **Hopsworks Feature Store**
- **Project**: `aqi_maazkhan`
- **Feature Group**: `london_air_quality_6h v1`
- **Feature View**: `london_air_quality_6h_view`
- **Records**: 9,192 samples
- **Features**: 138 columns
- **URL**: https://c.app.hopsworks.ai:443/p/1252518

### **Data Sources**
- **Open-Meteo Air Quality API**: Free, no authentication
- **Open-Meteo Weather API**: Free, no authentication
- **Location**: London, UK (51.5074°N, 0.1278°W)

---

## 🎯 Key Achievements

1. ✅ **Complete End-to-End Pipeline**: From raw data to predictions
2. ✅ **Cloud Integration**: Hopsworks Feature Store + Model Registry
3. ✅ **Free Data Sources**: No API costs for data collection
4. ✅ **Production-Ready**: Models saved and versioned in cloud
5. ✅ **Interactive Dashboard**: Real-time monitoring and analysis
6. ✅ **Feature Engineering**: 138 sophisticated time-series features
7. ✅ **Model Comparison**: XGBoost vs Random Forest analysis
8. ✅ **Clean Codebase**: Simplified, well-documented code

---

## 🔮 Future Enhancements

- **Real-time Predictions**: Live AQI forecasting
- **Multiple Cities**: Expand beyond London
- **Advanced Models**: LSTM, Transformer models
- **Alert System**: Air quality warnings
- **Mobile App**: React Native dashboard
- **API Endpoint**: REST API for predictions

---

**Project Status**: ✅ **COMPLETED & FUNCTIONAL**

*Last Updated: October 16, 2025*
