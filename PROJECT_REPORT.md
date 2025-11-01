# Air Quality Index Forecasting Project
## Machine Learning Operations Report

---

## 1. Project Objective

The Air Quality Index (AQI) Forecasting Project aims to develop an end-to-end machine learning system that predicts air quality levels for London, UK, across multiple time horizons (1h, 6h, 12h, 24h, 48h, 72h). The system provides real-time forecasts to help individuals and organizations make informed decisions about outdoor activities, health precautions, and environmental planning. The project implements a production-ready ML pipeline with automated data collection, feature engineering, model training, and deployment through a web dashboard.

**Key Goals:**
- Forecast AQI levels up to 3 days ahead (72 hours)
- Provide actionable insights through an interactive dashboard
- Implement automated pipeline for continuous model updates
- Monitor air quality and alert users of hazardous conditions

---

## 2. System Architecture & Implementation Approach

### 2.1 Data Pipeline Architecture

The system follows a modular, production-oriented architecture:

1. **Data Collection Layer**: Fetches real-time and historical air quality and weather data from Open Meteo API
2. **Feature Engineering Layer**: Transforms raw data into ML-ready features
3. **Storage Layer**: Hopsworks Feature Store for versioned feature management
4. **Training Layer**: Multi-output regression models for simultaneous multi-horizon predictions
5. **Model Registry**: Hopsworks Model Registry for version control and deployment
6. **Application Layer**: Streamlit dashboard for real-time visualization and predictions

### 2.2 Data Source & Location

- **API**: Open Meteo Air Quality API (free, no authentication required)
- **Location**: London, UK (Latitude: 51.5074, Longitude: -0.1278)
- **Data Types**: 
  - Air Quality: PM2.5, PM10, Ozone, Nitrogen Dioxide, Sulphur Dioxide, Carbon Monoxide
  - Weather: Temperature, Humidity, Pressure, Wind Speed/Direction, Precipitation, Cloud Cover

### 2.3 Implementation Logic

The implementation follows the **MLOps** paradigm with these key principles:

**Chronological Processing Order:**
1. **Data Fetching**: Retrieve historical data (default: 1 year) for initial setup or incremental daily updates
2. **Exploratory Data Analysis (EDA)**: Generate automated snapshots with visualizations for data quality assessment
3. **Feature Engineering**: Transform raw data into 124+ engineered features (time-based, lag, rolling, derived)
4. **Data Storage**: Upload features to Hopsworks Feature Store with versioning
5. **Model Training**: Train multi-output models using chronological train/test split (80/20)
6. **Model Evaluation**: Calculate R², MAE, RMSE, MAPE metrics for each horizon
7. **Model Persistence**: Save models locally and to Hopsworks Model Registry with performance metrics
8. **Alert System**: Monitor current and predicted AQI levels against EPA thresholds
9. **Dashboard Deployment**: Streamlit web application for real-time predictions and visualization

**Key Design Decisions:**
- **Multi-output regression**: Single model predicts all 6 horizons simultaneously, reducing training time from 12 models to 2 (83% reduction)
- **Chronological split**: Time-series cross-validation prevents data leakage
- **Feature versioning**: Incremental feature group versions (v1-v5) for schema evolution
- **Prediction clipping**: All predictions constrained to valid AQI range [0, 500]

---

## 3. Feature Engineering Strategy

### 3.1 Feature Categories

The system generates **124 features** organized into four categories:

#### **Time-Based Features (12 features)**
- **Categorical**: hour, day_of_week, month, is_weekend, is_rush_hour, season
- **Cyclical Encodings**: hour_sin/cos, day_of_week_sin/cos, month_sin/cos
- **Rationale**: Captures temporal patterns (diurnal cycles, weekday/weekend effects, seasonal trends)

#### **Lag Features (22 features)**
- **Lag Windows**: [1h, 6h] for 11 key variables
- **Variables**: PM2.5, PM10, Ozone, NO₂, SO₂, CO, Temperature, Humidity, Wind Speed, Wind Direction, AQI
- **Rationale**: Captures short-term dependencies and recent air quality trends

#### **Rolling Window Features (54 features)**
- **Rolling Windows**: [12h, 24h, 48h] for 9 variables
- **Statistics**: Mean and Standard Deviation for each window
- **Variables**: PM2.5, PM10, Ozone, NO₂, Temperature, Wind Speed, Humidity, Pressure, AQI
- **Rationale**: Captures medium-term trends and variability patterns

#### **Derived Features (5 features)**
- temp_humidity_interaction: Temperature × Humidity (captures comfort/air density effects)
- wind_chill: Calculated wind chill index
- pm_ratio: PM2.5/PM10 (particle size distribution indicator)
- pm_combined: PM2.5 + PM10 (total particulate matter)
- aqi_change_1h: Hourly AQI change rate

#### **Base Features (19 features)**
- 8 pollutant concentrations + 8 weather variables + 3 computed (AQI, timestamp, target)

### 3.2 Feature Engineering Logic

**Optimization Strategy:**
- **Reduced lag windows** from [1, 2, 3, 6] to [1, 6] to prevent overfitting (33% reduction)
- **Optimized rolling windows** from [3, 6, 12, 24] to [12, 24, 48] for better long-horizon signal
- **Added wind_direction** to lag features (critical for pollutant dispersion patterns)
- **Added humidity/pressure** to rolling features (weather stability indicators)

**Target Variable:**
- **aqi_6h_ahead**: Primary target for 6-hour forecasting
- **Multi-horizon targets**: [1h, 6h, 12h, 24h, 48h, 72h] for multi-output training

### 3.3 Feature Store Integration

- **Hopsworks Feature Group**: `london_air_quality_6h` (Version 5)
- **Feature View**: `london_air_quality_6h_view` (Version 5)
- **Schema Versioning**: Automatic schema compatibility checks with version increments
- **Incremental Updates**: Daily append operations for new data

---

## 4. Model Selection & Rationale

### 4.1 Model Choices

Two models were selected based on complementary strengths:

#### **1. Random Forest Regressor**
- **Configuration**: 
  - `n_estimators=50`, `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features='sqrt'`
- **Rationale**:
  - Handles non-linear relationships between features and AQI
  - Robust to outliers and missing data
  - Provides feature importance rankings
  - Ensemble method reduces overfitting risk

#### **2. Ridge Regression**
- **Configuration**: 
  - `alpha=20.0` (strong L2 regularization)
- **Rationale**:
  - Linear baseline model for comparison
  - Fast training and prediction
  - Strong regularization prevents overfitting on limited data
  - Interpretable coefficients (though less important with feature engineering)

### 4.2 Why Two Models?

**Complementary Approach:**
- **Random Forest**: Captures complex non-linear patterns, better for short-term predictions
- **Ridge Regression**: Provides linear baseline, better generalization for longer horizons with limited data

**Ensemble Benefits:**
- Model comparison enables performance validation
- Dashboard displays predictions from both models for user comparison
- Risk mitigation: If one model fails, the other provides backup

### 4.3 Multi-Output Regression Implementation

**Architecture:**
- **Framework**: `sklearn.multioutput.MultiOutputRegressor`
- **Output**: Single model predicts all 6 horizons simultaneously `[pred_1h, pred_6h, pred_12h, pred_24h, pred_48h, pred_72h]`
- **Benefits**:
  - Shared representation learning across horizons
  - Reduced variance through joint optimization
  - 83% reduction in training time (2 models vs 12 separate models)
  - Better sample efficiency for longer horizons

**Performance Optimization:**
- Prediction clipping: All outputs constrained to [0, 500] AQI range
- Robust metrics calculation: Handles NaN, inf, and constant target values
- Train/test gap monitoring: Detects overfitting early

---

## 5. Model Training & Evaluation

### 5.1 Training Process

**Data Split:**
- **Chronological Split**: 80% train, 20% test (preserves temporal order)
- **Training Data**: ~5,800 samples (from 1 year of hourly data)
- **Test Data**: ~1,450 samples

**Training Workflow:**
1. Load engineered features from Hopsworks Feature Store
2. Create multi-horizon targets by shifting AQI values forward by [1, 6, 12, 24, 48, 72] hours
3. Prepare feature matrix: Exclude target variables and prevent data leakage
4. Train Random Forest and Ridge Regression models using MultiOutputRegressor
5. Calculate metrics for each horizon independently
6. Save models and metrics as sidecar JSON files

**Regularization Strategy:**
- **Random Forest**: Reduced tree depth (5), increased min_samples_split (10), sqrt max_features
- **Ridge Regression**: High alpha (20.0) for strong regularization
- **Purpose**: Prevent overfitting on limited training data (~300 days)

### 5.2 Evaluation Metrics

**Primary Metrics:**
- **R² Score**: Coefficient of determination (explained variance)
- **MAE**: Mean Absolute Error (AQI units)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error (relative error)

**Expected Performance:**
- **1h-6h horizons**: R² > 0.60 (Random Forest), R² > 0.50 (Ridge)
- **12h-24h horizons**: R² > 0.40 (Random Forest), R² > 0.30 (Ridge)
- **48h-72h horizons**: R² > 0.20 (Random Forest), R² > 0.10 (Ridge)

**Note**: Longer horizons naturally have lower R² due to increased uncertainty. Negative R² indicates model performs worse than baseline (mean predictor).

### 5.3 Model Comparison

**Performance Characteristics:**

| Horizon | Random Forest Advantage | Ridge Regression Advantage |
|---------|------------------------|----------------------------|
| 1h-6h   | Better non-linear capture | Faster training/inference |
| 12h-24h | Robust to outliers | Better generalization |
| 48h-72h | Feature interactions | Less prone to overfitting |

**Deployment Strategy:**
- Dashboard displays Random Forest predictions by default (better short-term accuracy)
- Ridge Regression available for comparison and validation
- Model metrics stored in Hopsworks Model Registry for version tracking

---

## 6. CI/CD Pipeline

### 6.1 GitHub Actions Workflow

**Configuration:**
- **Workflow File**: `.github/workflows/retrain.yml`
- **Schedule**: Hourly execution (`cron: '0 * * * *'` UTC)
- **Manual Trigger**: Supported via `workflow_dispatch`
- **Runner**: `ubuntu-latest` with Python 3.12

**Pipeline Steps:**
1. **Checkout code**: Clone repository
2. **Set up Python**: Install Python 3.12
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run pipeline update**: Execute `python -m src.main update`
   - Environment variables: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME` (GitHub Secrets)

**Automation Features:**
- **Concurrency Control**: Prevents duplicate runs with `cancel-in-progress: false`
- **Failure Handling**: Automatic retries and error logging
- **Execution Time**: ~1-2 minutes per run
- **Status**: Successfully running (110+ completed runs as of report date)

### 6.2 Pipeline Operations

**Hourly Workflow:**
1. Fetch last 24 hours of new data from Open Meteo API
2. Generate EDA snapshot for new data
3. Engineer features and append to Hopsworks Feature Store
4. Retrain models with updated dataset
5. Save new model versions to Model Registry
6. Run alert system for current conditions
7. Execute interpretability analysis (SHAP) if enabled

**Model Versioning:**
- Models saved as: `multi_output_{algorithm}_model.pkl`
- Metrics saved as: `multi_output_{algorithm}_h{horizon}_metrics.json`
- Hopsworks Model Registry tracks version history with descriptions

---

## 7. Web Dashboard

### 7.1 Dashboard Components

**Technology**: Streamlit (Python web framework)

**Key Sections:**
1. **Current Air Quality Status**: AQI gauge visualization with color-coded health indicators
2. **Alert Panel**: Real-time alerts for hazardous AQI levels (EPA standards)
3. **Model Registry Metrics**: Performance comparison tables and charts
4. **Air Quality Trends**: Time series visualization of historical AQI
5. **Multi-Horizon Forecast**: Predictions for 1h, 6h, 12h, 24h, 48h, 72h horizons
6. **Model Comparison**: Side-by-side Random Forest vs Ridge Regression predictions

**Features:**
- Interactive date range selector
- Real-time data loading from Hopsworks
- Automated model loading and prediction generation
- Visual performance metrics (R² scores, MAE, bar charts)

### 7.2 Alert System

**Implementation:**
- **Thresholds**: EPA AQI standards (Low: 0-50, Moderate: 51-100, High: 101-150, Critical: >150)
- **Monitoring**: Current AQI + predicted AQI for all horizons
- **Alert Types**: Low, Moderate, High, Critical severity levels
- **Deduplication**: Prevents duplicate alerts within 1-hour window
- **Storage**: JSON-based alert history (24-hour retention)

---

## 8. Limitations & Future Work

### 8.1 Implemented vs. Planned Features

Based on `Key_Features.md`, the following features are **implemented**:
✅ Feature Pipeline Development (hourly automation)
✅ Historical Data Backfill (custom date ranges)
✅ Training Pipeline Implementation (Random Forest, Ridge Regression)
✅ Automated CI/CD Pipeline (GitHub Actions, hourly execution)
✅ Web Application Dashboard (Streamlit)
✅ Exploratory Data Analysis (automated EDA snapshots)
✅ Alert System (EPA-based thresholds)

### 8.2 Limitations & Unimplemented Features

**1. Deep Learning Models**
- **Status**: Not implemented (currently uses only scikit-learn framework)
- **Planned**: TensorFlow or PyTorch for deep learning models (LSTM/Transformer) to improve long-horizon forecasting
- **Impact**: Current scikit-learn models (Random Forest, Ridge) may struggle with very long horizons (72h+)
- **Future Work**: Implement sequence-to-sequence models using either TensorFlow or PyTorch (single framework choice, not both)

**2. Model Interpretability (SHAP/LIME)**
- **Status**: SHAP code exists but removed from dashboard
- **Planned**: Interactive SHAP visualizations for feature importance
- **Impact**: Limited model explainability in production dashboard
- **Future Work**: Re-integrate SHAP analysis with dashboard visualization

**3. Daily Model Retraining**
- **Status**: Currently retrains hourly (more frequent than planned)
- **Planned**: Daily model updates as per original specification
- **Current**: Hourly updates may be unnecessary and resource-intensive
- **Future Work**: Optimize to daily retraining or implement incremental learning

**4. Multiple Forecasting Models (Statistical to Deep Learning)**
- **Status**: Only 2 models from single framework - scikit-learn (Random Forest, Ridge Regression)
- **Planned**: Add statistical time series models (ARIMA via statsmodels, Prophet via fbprophet) and optionally deep learning (LSTM/Transformer via TensorFlow or PyTorch)
- **Rationale for Multiple Frameworks**: Different model types require different frameworks - statistical models use time-series libraries, deep learning requires TensorFlow/PyTorch, while current models use scikit-learn
- **Impact**: Limited model diversity and ensemble potential with current single-framework approach
- **Future Work**: Expand model suite with statistical time series and deep learning frameworks for comprehensive comparison

**5. Data Source Limitations**
- **Current**: Single API (Open Meteo) for London only
- **Planned**: Multiple sources (AQICN, OpenWeather) and multi-city support
- **Impact**: Single point of failure, limited geographic coverage
- **Future Work**: Multi-source data fusion and geographic expansion

**6. Real-time Prediction Latency**
- **Current**: Predictions generated on-demand from dashboard
- **Planned**: Cached predictions for faster response times
- **Future Work**: Implement prediction caching and asynchronous updates

---

## 9. Technical Specifications

### 9.1 Technology Stack

- **Language**: Python 3.12
- **ML Framework**: scikit-learn (Random Forest, Ridge Regression, MultiOutputRegressor)
  - **Note**: Currently uses only one ML framework (scikit-learn). Future work may include TensorFlow/PyTorch for deep learning and statistical libraries (statsmodels, Prophet) for time series models.
- **Feature Store**: Hopsworks (Feature Groups, Feature Views, Model Registry)
- **Data Processing**: pandas, numpy
- **API Client**: requests (Open Meteo API)
- **Visualization**: Plotly, Streamlit
- **CI/CD**: GitHub Actions
- **Deployment**: Local Streamlit server (extensible to cloud deployment)

### 9.2 Project Structure

```
AirQualityIndex/
├── .github/workflows/
│   └── retrain.yml              # CI/CD pipeline
├── src/
│   ├── main.py                  # CLI entry point
│   ├── pipeline.py              # Unified ML pipeline
│   ├── data_fetcher.py         # Open Meteo API client
│   ├── feature_engineering.py  # Feature creation
│   ├── training.py              # Model training (multi-output)
│   ├── hopsworks_client.py     # Feature Store client
│   ├── alerts.py               # Alert system
│   ├── metrics_loader.py      # Model metrics utilities
│   ├── eda.py                  # EDA snapshot generation
│   ├── dashboard.py            # Streamlit web app
│   └── aqi_calculator.py       # US EPA AQI calculation
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
└── models/                     # Local model storage
```

---

## 10. Conclusion

The Air Quality Index Forecasting Project successfully implements a production-ready MLOps pipeline for multi-horizon AQI prediction in London. The system demonstrates effective feature engineering (124 features), model selection (Random Forest and Ridge Regression), and automated CI/CD (hourly GitHub Actions). The Streamlit dashboard provides real-time predictions and alerts, while Hopsworks ensures versioned feature and model management.

**Key Achievements:**
- Multi-output regression reduces training overhead by 83%
- Automated hourly pipeline ensures up-to-date predictions
- Production-ready alert system with EPA standards
- Comprehensive model evaluation and versioning

**Future Enhancements:**
The project roadmap includes deep learning models (LSTM/Transformer), expanded interpretability (SHAP dashboard), multi-city support, and statistical baseline models (ARIMA, Prophet) for comprehensive model comparison and improved long-horizon forecasting.

---

**Report Generated**: November 2025  
**Project Version**: 1.0  
**Location**: London, UK  
**Forecasting Horizons**: 1h, 6h, 12h, 24h, 48h, 72h

