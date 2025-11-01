import pandas as pd
import numpy as np
import logging
import os
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
            
            # Step 2: Generate EDA snapshot
            logger.info("Generating EDA snapshot...")
            from src.eda import EDASnapshot
            eda = EDASnapshot()
            eda_results = eda.generate_eda_report(df, "initial_setup")
            
            if eda_results['status'] == 'SUCCESS':
                artifacts_count = len(eda_results.get('artifacts', {}))
                logger.info(f"EDA snapshot generated with {artifacts_count} visualizations")
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
            
            # Step 5: Train multi-output models (for all horizons)
            logger.info("Training multi-output models for all horizons...")
            X_multi, Y_multi = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_features)
            if X_multi is not None and Y_multi is not None:
                horizon_results = self.multi_horizon_forecaster.train_multi_horizon_models(X_multi, Y_multi)
                # Save multi-horizon models locally and to Hopsworks
                try:
                    import joblib, os
                    os.makedirs("models", exist_ok=True)
                    
                    # Multi-output models: one model per algorithm (not per horizon)
                    for model_name, result in horizon_results.items():
                        model = result['model']
                        test_metrics_by_horizon = result['test_metrics_by_horizon']
                        
                        # Save the multi-output model (predicts all horizons)
                        model_path = f"models/multi_output_{model_name}_model.pkl"
                        joblib.dump(model, model_path)
                        logger.info(f"Saved multi-output {model_name} model to {model_path}")
                        
                        # Save metrics for each horizon
                        for horizon in result['horizons']:
                            if horizon in test_metrics_by_horizon:
                                metrics = test_metrics_by_horizon[horizon]
                                metrics_path = f"models/multi_output_{model_name}_h{horizon}_metrics.json"
                                with open(metrics_path, 'w') as f:
                                    json.dump({
                                        'model_name': f"multi_output_{model_name}",
                                        'horizon': horizon,
                                        'algorithm': model_name,
                                        'metrics': metrics,
                                        'training_timestamp': datetime.now().isoformat(),
                                        'model_path': model_path,
                                        'version': '1.0'
                                    }, f, indent=2)
                                
                                logger.info(f"Saved metrics for {model_name} {horizon}h to {metrics_path}")
                        
                        # Save aggregate metrics to Hopsworks (using 6h horizon as representative)
                        representative_metrics = test_metrics_by_horizon.get(6, test_metrics_by_horizon[list(test_metrics_by_horizon.keys())[0]])
                        description = f"Multi-output AQI forecasting model ({model_name}) for all horizons: {result['horizons']}"
                        
                        self.client.save_model(
                            model=model_path,
                            model_name=f"{config.HOPSWORKS_MODEL_NAME}_multi_output_{model_name}",
                            metrics=representative_metrics,
                            description=description
                        )
                except Exception as e:
                    logger.error(f"Failed to save multi-horizon models: {str(e)}")
            
            # Step 7: Run alerts and interpretability
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
            
            # Step 2: Generate EDA snapshot for new data
            logger.info("Generating EDA snapshot for new data...")
            from src.eda import EDASnapshot
            eda = EDASnapshot()
            eda_results = eda.generate_eda_report(df_new, "daily_update")
            
            if eda_results['status'] == 'SUCCESS':
                artifacts_count = len(eda_results.get('artifacts', {}))
                logger.info(f"EDA snapshot generated with {artifacts_count} visualizations")
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
            
            # Step 5: Get updated data and retrain multi-output models
            logger.info("Retraining multi-output models with updated data...")
            df_updated = self.client.get_feature_data()
            
            if df_updated is None or df_updated.empty:
                logger.error("No updated data available from Hopsworks")
                return False
            
            logger.info(f"Retraining with {len(df_updated)} total records")
            
            # Retrain multi-output models
            X_multi, Y_multi = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_updated)
            if X_multi is not None and Y_multi is not None:
                horizon_results = self.multi_horizon_forecaster.train_multi_horizon_models(X_multi, Y_multi)
                try:
                    import joblib, os
                    os.makedirs("models", exist_ok=True)
                    
                    # Save multi-output models (one per algorithm)
                    for model_name, result in horizon_results.items():
                        model = result['model']
                        test_metrics_by_horizon = result['test_metrics_by_horizon']
                        
                        # Save the multi-output model (predicts all horizons)
                        model_path = f"models/multi_output_{model_name}_model.pkl"
                        joblib.dump(model, model_path)
                        logger.info(f"Saved multi-output {model_name} model to {model_path}")
                        
                        # Save metrics for each horizon
                        for horizon in result['horizons']:
                            if horizon in test_metrics_by_horizon:
                                metrics = test_metrics_by_horizon[horizon]
                                metrics_path = f"models/multi_output_{model_name}_h{horizon}_metrics.json"
                                with open(metrics_path, 'w') as f:
                                    json.dump({
                                        'model_name': f"multi_output_{model_name}",
                                        'horizon': horizon,
                                        'algorithm': model_name,
                                        'metrics': metrics,
                                        'training_timestamp': datetime.now().isoformat(),
                                        'model_path': model_path,
                                        'version': '1.0',
                                        'update_type': 'daily'
                                    }, f, indent=2)
                                
                                logger.info(f"Saved metrics for {model_name} {horizon}h to {metrics_path}")
                        
                        # Use 6h metrics as representative for Hopsworks
                        representative_metrics = test_metrics_by_horizon.get(6, test_metrics_by_horizon[list(test_metrics_by_horizon.keys())[0]])
                        
                        self.client.save_model(
                            model=model_path,
                            model_name=f"{config.HOPSWORKS_MODEL_NAME}_multi_output_{model_name}",
                            metrics=representative_metrics,
                            description=f"Multi-output AQI forecasting model ({model_name}) for all horizons (daily update)"
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
            
            # Load multi-output models for alert system
            import joblib
            import os
            models_for_alerts = {}
            for model_key in ['random_forest', 'ridge_regression']:
                model_path = f"models/multi_output_{model_key}_model.pkl"
                if os.path.exists(model_path):
                    try:
                        models_for_alerts[model_key] = joblib.load(model_path)
                    except Exception as e:
                        logger.warning(f"Failed to load {model_key} for alerts: {str(e)}")
            
            if not models_for_alerts:
                logger.warning("No trained multi-output models available for alert system")
                return True
            
            # Run alert check
            alert_results = alert_system.run_alert_check(df_data, models_for_alerts)
            
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
            
            # Load multi-output models for SHAP analysis
            import joblib
            import os
            models_for_shap = {}
            for model_key in ['random_forest', 'ridge_regression']:
                model_path = f"models/multi_output_{model_key}_model.pkl"
                if os.path.exists(model_path):
                    try:
                        models_for_shap[model_key] = joblib.load(model_path)
                    except Exception as e:
                        logger.warning(f"Failed to load {model_key} for SHAP: {str(e)}")
            
            if not models_for_shap:
                logger.warning("No trained multi-output models available for interpretability analysis")
                return True
            
            # Get training data for SHAP analysis
            df_data = self.client.get_feature_data()
            if df_data is None or df_data.empty:
                logger.warning("No data available for interpretability analysis")
                return True
            
            # Prepare multi-output data (same as training)
            X_multi, Y_multi = self.multi_horizon_forecaster.prepare_multi_horizon_data(df_data)
            if X_multi is None or Y_multi is None or X_multi.empty:
                logger.warning("Failed to prepare multi-output data for SHAP analysis")
                return True
            
            # Chronological split
            split_idx = int(len(X_multi) * config.TRAIN_TEST_SPLIT)
            X_train, X_test = X_multi.iloc[:split_idx], X_multi.iloc[split_idx:]
            Y_train, Y_test = Y_multi.iloc[:split_idx], Y_multi.iloc[split_idx:]
            
            # Use 6h horizon for SHAP analysis (index 1 in horizons [1, 6, 12, 24, 48, 72])
            y_train_shap = Y_train.iloc[:, 1]  # 6h horizon
            y_test_shap = Y_test.iloc[:, 1]
            
            logger.info(f"Running SHAP analysis on {len(models_for_shap)} multi-output models")
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Run SHAP analysis on each model (using 6h horizon predictions)
            shap_results = {}
            for model_name, model in models_for_shap.items():
                logger.info(f"Analyzing {model_name} multi-output model (6h horizon)...")
                # For multi-output models, we analyze the 6h horizon (index 1)
                # Create a wrapper that extracts 6h prediction
                class SingleOutputWrapper:
                    def __init__(self, multi_output_model, horizon_idx=1):
                        self.model = multi_output_model
                        self.horizon_idx = horizon_idx
                    
                    def predict(self, X):
                        pred_all = self.model.predict(X)
                        return pred_all[:, self.horizon_idx] if len(pred_all.shape) > 1 else [pred_all[self.horizon_idx]]
                
                wrapped_model = SingleOutputWrapper(model, horizon_idx=1)  # 6h horizon
                
                result = analyzer.analyze_model(
                    model=wrapped_model,
                    X_train=X_train,
                    X_test=X_test,
                    model_name=f"{model_name}_6h",
                    feature_names=X_train.columns.tolist()
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
