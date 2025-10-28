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
    
    def __init__(self, mode='setup', date_params=None):
        self.mode = mode
        self.date_params = date_params or {}
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
        """Initial setup: Fetch data for specified date range, train models, save to Hopsworks"""
        try:
            logger.info("=== INITIAL SETUP MODE ===")
            
            # Step 1: Determine date range
            if 'start_date' in self.date_params and 'end_date' in self.date_params:
                start_date_str = self.date_params['start_date']
                end_date_str = self.date_params['end_date']
                
                # Validate date format
                try:
                    datetime.strptime(start_date_str, '%Y-%m-%d')
                    datetime.strptime(end_date_str, '%Y-%m-%d')
                except ValueError as e:
                    logger.error(f"Invalid date format: {str(e)}. Use YYYY-MM-DD format.")
                    return False
                
                logger.info(f"Using custom date range: {start_date_str} to {end_date_str}")
            else:
                # Default: 1 year of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                logger.info(f"Using default date range (1 year): {start_date_str} to {end_date_str}")
            
            logger.info(f"Fetching historical data from {start_date_str} to {end_date_str}")
            
            df = self.data_fetcher.fetch_combined_historical_data(
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if df is None or df.empty:
                logger.error("Failed to fetch historical data")
                return False
            
            logger.info(f"Fetched {len(df)} records for initial setup")
            
            # Step 2: Validate data quality
            logger.info("Validating data quality...")
            from src.validation import DataValidator
            validator = DataValidator()
            validation_results = validator.validate_data(df, "initial_setup")
            
            if validation_results['overall_status'] == 'FAIL':
                logger.error("Data validation failed - critical issues detected")
                logger.error(f"Validation issues: {validation_results.get('issues', [])}")
                return False
            elif validation_results['overall_status'] == 'WARN':
                logger.warning("Data validation warnings detected")
                logger.warning(f"Validation issues: {validation_results.get('issues', [])}")
            
            logger.info(f"Data validation completed: {validation_results['overall_status']}")
            
            # Step 3: Generate EDA snapshot
            logger.info("Generating EDA snapshot...")
            from src.eda import EDASnapshot
            eda = EDASnapshot()
            eda_results = eda.generate_eda_report(df, "initial_setup")
            
            if eda_results['status'] == 'SUCCESS':
                logger.info(f"EDA snapshot generated: {eda_results.get('artifacts', {}).get('html_report', 'Unknown')}")
            else:
                logger.warning(f"EDA snapshot generation failed: {eda_results.get('error', 'Unknown error')}")
            
            # Step 4: Engineer features
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
            
            # Prepare training data for 6h ahead target (avoid leakage)
            # Use 'aqi_6h_ahead' as target and exclude non-lagged AQI-derived features
            numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and direct AQI columns; keep only lag-based AQI features
            numeric_features = [
                c for c in numeric_features
                if c != 'aqi_6h_ahead' and c != 'aqi' and not (c.startswith('aqi_') and not c.startswith('aqi_lag_'))
            ]
            
            X_all = df_features[numeric_features]
            y_all = df_features['aqi_6h_ahead']
            
            # Chronological split to avoid leakage
            split_idx = int(len(X_all) * config.TRAIN_TEST_SPLIT)
            X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
            y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
            
            self.forecaster.train_from_dataframe(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
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
                
                # Save metrics as sidecar JSON file
                metrics_path = f"models/{name}_metrics.json"
                import json
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'model_name': name,
                        'metrics': metrics,
                        'training_timestamp': datetime.now().isoformat(),
                        'model_path': model_path,
                        'version': '1.0'
                    }, f, indent=2)
                
                logger.info(f"Saved metrics for {name} to {metrics_path}")
                
                # Create enhanced description with metrics reference
                description = f"Initial setup for {name} model. Metrics: {metrics_path}"
                
                # Save to Hopsworks
                self.client.save_model(
                    model=model_path,
                    model_name=f"{config.HOPSWORKS_MODEL_NAME}_{name}",
                    metrics=metrics,
                    description=description
                )
            
            # Step 7: Train multi-horizon models
            logger.info("Training multi-horizon models...")
            horizon_data = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_features)
            if horizon_data:
                horizon_results = self.multi_horizon_forecaster.train_multi_horizon_models(horizon_data)
                # Save multi-horizon models locally and to Hopsworks
                try:
                    import joblib, os
                    os.makedirs("models", exist_ok=True)
                    for horizon, results in horizon_results.items():
                        for name, result in results.items():
                            model = result['model']
                            metrics = result['test_metrics']
                            model_path = f"models/h{horizon}_{name}_model.pkl"
                            joblib.dump(model, model_path)
                            
                            # Save metrics as sidecar JSON file
                            metrics_path = f"models/h{horizon}_{name}_metrics.json"
                            with open(metrics_path, 'w') as f:
                                json.dump({
                                    'model_name': f"h{horizon}_{name}",
                                    'horizon': horizon,
                                    'algorithm': name,
                                    'metrics': metrics,
                                    'training_timestamp': datetime.now().isoformat(),
                                    'model_path': model_path,
                                    'version': '1.0'
                                }, f, indent=2)
                            
                            logger.info(f"Saved metrics for h{horizon}_{name} to {metrics_path}")
                            
                            # Create enhanced description with metrics reference
                            description = f"{horizon}-hour AQI forecasting model using {name}. Metrics: {metrics_path}"
                            
                            self.client.save_model(
                                model=model_path,
                                model_name=f"{config.HOPSWORKS_MODEL_NAME}_h{horizon}_{name}",
                                metrics=metrics,
                                description=description
                            )
                except Exception as e:
                    logger.error(f"Failed to save multi-horizon models: {str(e)}")
            
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
            
            # Step 2: Validate new data quality
            logger.info("Validating new data quality...")
            from src.validation import DataValidator
            validator = DataValidator()
            validation_results = validator.validate_data(df_new, "daily_update")
            
            if validation_results['overall_status'] == 'FAIL':
                logger.error("New data validation failed - critical issues detected")
                logger.error(f"Validation issues: {validation_results.get('issues', [])}")
                return False
            elif validation_results['overall_status'] == 'WARN':
                logger.warning("New data validation warnings detected")
                logger.warning(f"Validation issues: {validation_results.get('issues', [])}")
            
            logger.info(f"New data validation completed: {validation_results['overall_status']}")
            
            # Step 3: Generate EDA snapshot for new data
            logger.info("Generating EDA snapshot for new data...")
            from src.eda import EDASnapshot
            eda = EDASnapshot()
            eda_results = eda.generate_eda_report(df_new, "daily_update")
            
            if eda_results['status'] == 'SUCCESS':
                logger.info(f"EDA snapshot generated: {eda_results.get('artifacts', {}).get('html_report', 'Unknown')}")
            else:
                logger.warning(f"EDA snapshot generation failed: {eda_results.get('error', 'Unknown error')}")
            
            # Step 4: Engineer features for new data
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
            
            # Retrain models with chronological split and 6h ahead target
            numeric_features = df_updated.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features
                if c != 'aqi_6h_ahead' and c != 'aqi' and not (c.startswith('aqi_') and not c.startswith('aqi_lag_'))
            ]
            
            X_all = df_updated[numeric_features]
            y_all = df_updated['aqi_6h_ahead']
            
            split_idx = int(len(X_all) * config.TRAIN_TEST_SPLIT)
            X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
            y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
            
            self.forecaster.train_from_dataframe(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
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
                
                # Save metrics as sidecar JSON file
                metrics_path = f"models/{name}_metrics.json"
                import json
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'model_name': name,
                        'metrics': metrics,
                        'training_timestamp': datetime.now().isoformat(),
                        'model_path': model_path,
                        'version': '1.0',
                        'update_type': 'daily'
                    }, f, indent=2)
                
                logger.info(f"Saved updated metrics for {name} to {metrics_path}")
                
                # Create enhanced description with metrics reference
                description = f"Daily update for {name} model. Metrics: {metrics_path}"
                
                # Save to Hopsworks
                self.client.save_model(
                    model=model_path,
                    model_name=f"{config.HOPSWORKS_MODEL_NAME}_{name}",
                    metrics=metrics,
                    description=description
                )
            
            # Retrain multi-horizon models
            horizon_data = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_updated)
            if horizon_data:
                horizon_results = self.multi_horizon_forecaster.train_multi_horizon_models(horizon_data)
                try:
                    import joblib, os
                    os.makedirs("models", exist_ok=True)
                    for horizon, results in horizon_results.items():
                        for name, result in results.items():
                            model = result['model']
                            model_path = f"models/h{horizon}_{name}_model.pkl"
                            joblib.dump(model, model_path)
                            self.client.save_model(
                                model=model_path,
                                model_name=f"{config.HOPSWORKS_MODEL_NAME}_h{horizon}_{name}",
                                metrics=result['test_metrics'],
                                description=f"{horizon}-hour AQI forecasting model using {name} (daily update)"
                            )
                except Exception as e:
                    logger.error(f"Failed to save multi-horizon models: {str(e)}")
            
            # Step 6: Run alerts
            logger.info("Running alert system...")
            self.run_alert_system()
            
            logger.info("Daily update completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Daily update failed: {str(e)}")
            return False
    
    def run_alert_system(self) -> bool:
        """Run AQI alert system"""
        try:
            logger.info("Running AQI alert system...")
            
            # Import alert system
            from src.alerts import AQIAlertSystem
            
            alert_system = AQIAlertSystem()
            
            # Get current data for alert evaluation
            df_data = self.client.get_feature_data()
            if df_data is None or df_data.empty:
                logger.warning("No data available for alert system")
                return True
            
            # Get trained models
            if not hasattr(self.forecaster, 'models') or not self.forecaster.models:
                logger.warning("No trained models available for alert system")
                return True
            
            # Run alert check
            alert_results = alert_system.run_alert_check(df_data, self.forecaster.models)
            
            if alert_results.get('alerts'):
                logger.info(f"Alert system generated {len(alert_results['alerts'])} alerts")
                
                # Log critical alerts
                for alert in alert_results['alerts']:
                    if alert['severity'] == 'critical':
                        logger.warning(f"CRITICAL ALERT: {alert['message']}")
            else:
                logger.info("No new alerts generated")
            
            # Log alert summary
            summary = alert_results.get('summary', {})
            if summary:
                logger.info(f"Alert summary: {summary.get('total_alerts', 0)} alerts in last 24h")
            
            return True
                
        except Exception as e:
            logger.error(f"Alert system failed: {str(e)}")
            return False
    
    def run_interpretability_analysis(self) -> bool:
        """Run SHAP model interpretability analysis"""
        try:
            logger.info("Running SHAP interpretability analysis...")
            
            # Import SHAP analyzer
            from src.interpretability import SHAPAnalyzer
            
            analyzer = SHAPAnalyzer()
            
            # Get the latest trained models
            if not hasattr(self.forecaster, 'models') or not self.forecaster.models:
                logger.warning("No trained models available for interpretability analysis")
                return True
            
            # Get training data for SHAP analysis
            df_data = self.client.get_feature_data()
            if df_data is None or df_data.empty:
                logger.warning("No data available for interpretability analysis")
                return True
            
            # Prepare features (same logic as training)
            numeric_features = df_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [
                c for c in numeric_features
                if c != 'aqi_6h_ahead' and c != 'aqi' and not (c.startswith('aqi_') and not c.startswith('aqi_lag_'))
            ]
            
            X_all = df_data[numeric_features]
            y_all = df_data['aqi_6h_ahead']
            
            # Chronological split (same as training)
            split_idx = int(len(X_all) * config.TRAIN_TEST_SPLIT)
            X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
            y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
            
            logger.info(f"Running SHAP analysis on {len(self.forecaster.models)} models")
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Run SHAP analysis on each model
            shap_results = {}
            for model_name, model in self.forecaster.models.items():
                logger.info(f"Analyzing {model_name} model...")
                result = analyzer.analyze_model(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    model_name=model_name,
                    feature_names=numeric_features
                )
                
                if result:
                    shap_results[model_name] = result
                    logger.info(f"SHAP analysis completed for {model_name}")
                else:
                    logger.warning(f"SHAP analysis failed for {model_name}")
            
            # Save combined results
            if shap_results:
                import json
                combined_file = "artifacts/shap_analysis_summary.json"
                os.makedirs("artifacts", exist_ok=True)
                
                with open(combined_file, 'w') as f:
                    json.dump({
                        'analysis_timestamp': datetime.now().isoformat(),
                        'models_analyzed': list(shap_results.keys()),
                        'results': shap_results
                    }, f, indent=2)
                
                logger.info(f"SHAP analysis completed for {len(shap_results)} models")
                logger.info(f"Results saved to {combined_file}")
            else:
                logger.warning("No SHAP analysis results generated")
            
            return True
                
        except Exception as e:
            logger.error(f"Interpretability analysis failed: {str(e)}")
            return False


def run_unified_pipeline(mode='setup', date_params=None):
    """Main function to run unified pipeline"""
    try:
        logger.info(f"Starting unified pipeline in '{mode}' mode")
        
        pipeline = UnifiedPipeline(mode=mode, date_params=date_params)
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
