import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional
import config
from src.aqi_calculator import calculate_aqi_for_dataframe

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.lag_hours = config.LAG_HOURS
        self.rolling_windows = config.ROLLING_WINDOWS
        self.forecast_horizon = config.FORECAST_HORIZON

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18]).astype(int)
        
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,
                                        3: 1, 4: 1, 5: 1,
                                        6: 2, 7: 2, 8: 2,
                                        9: 3, 10: 3, 11: 3})
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
            df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']
        
        if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
            df['wind_chill'] = 13.12 + 0.6215 * df['temperature_2m'] - 11.37 * (df['wind_speed_10m'] ** 0.16) + 0.3965 * df['temperature_2m'] * (df['wind_speed_10m'] ** 0.16)
        
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
            df['pm_combined'] = df['pm2_5'] + df['pm10']
        
        if 'aqi' in df.columns:
            df['aqi_change_1h'] = df['aqi'].diff()
        
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        lag_columns = ['pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide', 'sulphur_dioxide', 
                      'carbon_monoxide', 'temperature_2m', 'relative_humidity_2m', 
                      'wind_speed_10m', 'aqi']
        
        for col in lag_columns:
            if col in df.columns:
                for lag in self.lag_hours:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        rolling_columns = ['pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide', 
                          'temperature_2m', 'wind_speed_10m', 'aqi']
        
        for col in rolling_columns:
            if col in df.columns:
                for window in self.rolling_windows:
                    df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
        
        return df

    def engineer_features(self, df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            logger.info("Starting feature engineering...")
            
            df = df_raw.copy()
            
            # Calculate AQI first from pollutant data
            logger.info("Calculating AQI from pollutant concentrations...")
            df = calculate_aqi_for_dataframe(df)
            
            df = self.add_time_features(df)
            df = self.add_derived_features(df)
            
            df = self.add_lag_features(df)
            df = self.add_rolling_features(df)
            
            df['aqi_6h_ahead'] = df['aqi'].shift(-self.forecast_horizon)
            
            df = df.dropna()
            
            feature_breakdown = {
                'time_features': 12,
                'lag_features': len(self.lag_hours) * 10,
                'rolling_features': len(self.rolling_windows) * 2 * 7,
                'pollutant_features': len(config.POLLUTANT_FEATURES),
                'weather_features': len(config.WEATHER_FEATURES),
                'derived_features': 5
            }
            
            logger.info(f"Feature engineering completed")
            logger.info(f"Total features: {len(df.columns)}")
            logger.info(f"Training samples: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return None