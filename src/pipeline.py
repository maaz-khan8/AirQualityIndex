"""
Unified Pipeline for Air Quality Index Forecasting
Handles setup, updates, alerts, and interpretability in one place
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import config
from src.data_fetcher import OpenMeteoFetcher
from src.feature_engineering import FeatureEngineer
from src.training import AQIForecaster, MultiHorizonForecaster
from src.hopsworks_client import HopsworksClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedPipeline:
    """
    Unified pipeline that handles all operations based on mode:
    - setup: Initial setup with 1 year of data
    - update: Daily updates with new data
    - alerts: Run alert system
    - interpretability: Run SHAP analysis
    """
    
    def __init__(self, mode='setup'):
        self.mode = mode
        self.client = HopsworksClient()
        self.data_fetcher = OpenMeteoFetcher()
        self.feature_engineer = FeatureEngineer()
        self.forecaster = AQIForecaster()
        self.multi_horizon_forecaster = MultiHorizonForecaster()
        
    def run(self) -> bool:
        """Main entry point for all pipeline operations"""
        try:
            logger.info(f"Starting unified pipeline in '{self.mode}' mode")
            
            if self.mode == 'setup':
                return self.initial_setup()
            elif self.mode == 'update':
                return self.daily_update()
            elif self.mode == 'alerts':
                return self.run_alert_system()
            elif self.mode == 'interpretability':
                return self.run_interpretability_analysis()
            else:
                logger.error(f"Unknown mode: {self.mode}")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline failed in {self.mode} mode: {str(e)}")
            return False
    
    def initial_setup(self) -> bool:
        """Initial setup: Fetch 1 year of data, train models, save to Hopsworks"""
        try:
            logger.info("=== INITIAL SETUP MODE ===")
            logger.info("Fetching 1 year of historical data...")
            
            # Step 1: Fetch 1 year of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data from {start_date_str} to {end_date_str}")
            
            df = self.data_fetcher.fetch_combined_historical_data(
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if df is None or df.empty:
                logger.error("Failed to fetch historical data")
                return False
            
            logger.info(f"Fetched {len(df)} records for initial setup")
            
            # Step 2: Engineer features
            logger.info("Engineering features...")
            df_features = self.feature_engineer.engineer_features(df)
            
            if df_features is None or df_features.empty:
                logger.error("Feature engineering failed")
                return False
            
            logger.info(f"Engineered {len(df_features)} samples with {len(df_features.columns)} features")
            
            # Step 3: Connect to Hopsworks
            if not self.client.connect():
                logger.error("Failed to connect to Hopsworks")
                return False
            
            # Step 4: Upload to Hopsworks
            logger.info("Uploading data to Hopsworks...")
            fg = self.client.create_feature_group(df_features)
            if fg is None:
                logger.error("Failed to upload data to Hopsworks")
                return False
            
            # Step 5: Train models
            logger.info("Training models...")
            
            # Prepare training data - only numeric features
            numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
            if 'aqi' in numeric_features:
                numeric_features.remove('aqi')
            
            X = df_features[numeric_features]
            y = df_features['aqi']
            
            self.forecaster.train_from_dataframe(
                X_train=X,
                X_test=X,
                y_train=y,
                y_test=y
            )
            
            # Step 6: Save models to Hopsworks
            logger.info("Saving models to Hopsworks...")
            import joblib
            import os
            
            for name, model in self.forecaster.models.items():
                metrics = self.forecaster.results[name]['test_metrics']
                
                # Save model to file first
                model_path = f"models/{name}_model.pkl"
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)
                
                # Save to Hopsworks
                self.client.save_model(
                    model=model_path,
                    model_name=f"{config.HOPSWORKS_MODEL_NAME}_{name}",
                    metrics=metrics,
                    description=f"Initial setup for {name} model"
                )
            
            # Step 7: Train multi-horizon models
            logger.info("Training multi-horizon models...")
            horizon_data = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_features)
            if horizon_data:
                self.multi_horizon_forecaster.train_multi_horizon_models(horizon_data)
            
            # Step 8: Run alerts and interpretability
            logger.info("Running alert system...")
            self.run_alert_system()
            
            logger.info("Running interpretability analysis...")
            self.run_interpretability_analysis()
            
            logger.info("Initial setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Initial setup failed: {str(e)}")
            return False
    
    def daily_update(self) -> bool:
        """Daily update: Fetch 1 day of new data, append to Hopsworks, retrain models"""
        try:
            logger.info("=== DAILY UPDATE MODE ===")
            logger.info("Fetching 1 day of new data...")
            
            # Step 1: Fetch new data (last 1 day)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data from {start_date_str} to {end_date_str}")
            
            df_new = self.data_fetcher.fetch_combined_historical_data(
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if df_new is None or df_new.empty:
                logger.warning("No new data available, skipping update")
                return True
            
            logger.info(f"Fetched {len(df_new)} new records")
            
            # Step 2: Engineer features for new data
            logger.info("Engineering features for new data...")
            df_new_features = self.feature_engineer.engineer_features(df_new)
            
            if df_new_features is None or df_new_features.empty:
                logger.error("Feature engineering failed for new data")
                return False
            
            logger.info(f"Engineered {len(df_new_features)} new samples")
            
            # Step 3: Connect to Hopsworks
            if not self.client.connect():
                logger.error("Failed to connect to Hopsworks")
                return False
            
            # Step 4: Append new data to existing feature group
            logger.info("Appending new data to Hopsworks...")
            fg = self.client.create_feature_group(df_new_features)
            if fg is None:
                logger.error("Failed to append data to Hopsworks")
                return False
            
            logger.info(f"Successfully appended {len(df_new_features)} records to Hopsworks")
            
            # Step 5: Get updated data and retrain models
            logger.info("Retraining models with updated data...")
            df_updated = self.client.get_feature_data()
            
            if df_updated is None or df_updated.empty:
                logger.error("No updated data available from Hopsworks")
                return False
            
            logger.info(f"Retraining with {len(df_updated)} total records")
            
            # Retrain models
            # Prepare training data - only numeric features
            numeric_features = df_updated.select_dtypes(include=[np.number]).columns.tolist()
            if 'aqi' in numeric_features:
                numeric_features.remove('aqi')
            
            X = df_updated[numeric_features]
            y = df_updated['aqi']
            
            self.forecaster.train_from_dataframe(
                X_train=X,
                X_test=X,
                y_train=y,
                y_test=y
            )
            
            # Save updated models
            import joblib
            import os
            
            for name, model in self.forecaster.models.items():
                metrics = self.forecaster.results[name]['test_metrics']
                
                # Save model to file first
                model_path = f"models/{name}_model.pkl"
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)
                
                # Save to Hopsworks
                self.client.save_model(
                    model=model_path,
                    model_name=f"{config.HOPSWORKS_MODEL_NAME}_{name}",
                    metrics=metrics,
                    description=f"Daily update for {name} model"
                )
            
            # Retrain multi-horizon models
            horizon_data = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_updated)
            if horizon_data:
                self.multi_horizon_forecaster.train_multi_horizon_models(horizon_data)
            
            # Step 6: Run alerts
            logger.info("Running alert system...")
            self.run_alert_system()
            
            logger.info("Daily update completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Daily update failed: {str(e)}")
            return False
    
    def run_alert_system(self) -> bool:
        """Run AQI alert system (placeholder)"""
        try:
            logger.info("Alert system not implemented in unified pipeline")
            logger.info("Alert system: Skipped (module removed during refactoring)")
            return True
                
        except Exception as e:
            logger.error(f"Alert system failed: {str(e)}")
            return False
    
    def run_interpretability_analysis(self) -> bool:
        """Run model interpretability analysis (placeholder)"""
        try:
            logger.info("Interpretability analysis not implemented in unified pipeline")
            logger.info("Interpretability analysis: Skipped (module removed during refactoring)")
            return True
                
        except Exception as e:
            logger.error(f"Interpretability analysis failed: {str(e)}")
            return False


def run_unified_pipeline(mode='setup'):
    """Main function to run unified pipeline"""
    try:
        logger.info(f"Starting unified pipeline in '{mode}' mode")
        
        pipeline = UnifiedPipeline(mode=mode)
        success = pipeline.run()
        
        if success:
            logger.info(f"Unified pipeline completed successfully in '{mode}' mode")
        else:
            logger.error(f"Unified pipeline failed in '{mode}' mode")
            
        return success
        
    except Exception as e:
        logger.error(f"Unified pipeline failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'setup'
    run_unified_pipeline(mode)
