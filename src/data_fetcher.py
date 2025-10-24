import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from typing import Optional
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OpenMeteoFetcher:
    def __init__(self):
        self.location = config.LOCATION
        self.base_url_aq = config.OPENMETEO_AIR_QUALITY_URL
        self.base_url_weather = config.OPENMETEO_WEATHER_URL

    def fetch_air_quality_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            params = {
                'latitude': self.location['latitude'],
                'longitude': self.location['longitude'],
                'start_date': start_date,
                'end_date': end_date,
                'hourly': ','.join(config.POLLUTANT_FEATURES)
            }
            
            response = requests.get(self.base_url_aq, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'hourly' not in data:
                logger.error("No hourly data in response")
                return None
                
            df = pd.DataFrame(data['hourly'])
            
            if df.empty:
                logger.error("Empty DataFrame from air quality API")
                return None
    
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.drop('time', axis=1)
            
            logger.info(f"Fetched {len(df)} air quality records")
            return df
            
        except Exception as e:
            logger.error(f"Air quality fetch failed: {str(e)}")
            return None

    def fetch_weather_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            # For historical data, we need to use the historical weather API
            # The forecast API doesn't support historical data
            historical_url = "https://archive-api.open-meteo.com/v1/archive"
            
            params = {
                'latitude': self.location['latitude'],
                'longitude': self.location['longitude'],
                'start_date': start_date,
                'end_date': end_date,
                'hourly': ','.join(config.WEATHER_FEATURES)
            }
            
            response = requests.get(historical_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'hourly' not in data:
                logger.error("No hourly data in response")
                return None
            
            df = pd.DataFrame(data['hourly'])
            
            if df.empty:
                logger.error("Empty DataFrame from weather API")
                return None
            
            df['timestamp'] = pd.to_datetime(df['time'])
            df = df.drop('time', axis=1)
            
            logger.info(f"Fetched {len(df)} weather records")
            return df
            
        except Exception as e:
            logger.error(f"Weather fetch failed: {str(e)}")
            return None

    def fetch_combined_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            logger.info(f"Fetching data for {self.location['city']} from {start_date} to {end_date}")
            
            df_aq = self.fetch_air_quality_data(start_date, end_date)
            df_weather = self.fetch_weather_data(start_date, end_date)
            
            if df_aq is None or df_weather is None:
                logger.error("Failed to fetch one or both datasets")
                return None
            
            df_combined = pd.merge(df_aq, df_weather, on='timestamp', how='inner')
            
            df_combined['city'] = self.location['city']
            df_combined['latitude'] = self.location['latitude']
            df_combined['longitude'] = self.location['longitude']
            
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Combined dataset: {len(df_combined)} records, {len(df_combined.columns)} columns")
            return df_combined
            
        except Exception as e:
            logger.error(f"Combined fetch failed: {str(e)}")
            return None

    def get_data_summary(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            if df.empty:
                return None
            
            pollutants = {}
            weather = {}
            
            for col in df.columns:
                if col in ['timestamp', 'city', 'latitude', 'longitude']:
                    continue
                    
                if col in config.POLLUTANT_FEATURES:
                    pollutants[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                elif col in config.WEATHER_FEATURES:
                    weather[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
            
            return {
                'pollutants': pollutants,
                'weather': weather
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return None