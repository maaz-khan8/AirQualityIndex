import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
from typing import Dict, Tuple
import config
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


class AQIForecaster:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_names = []
        
        for name, params in config.MODELS_CONFIG.items():
            if name == 'random_forest':
                self.models[name] = RandomForestRegressor(**params)
            elif name == 'ridge_regression':
                self.models[name] = Ridge(**params)

    def _calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def train_from_hopsworks(self):
        try:
            client = HopsworksClient()
            if not client.connect():
                logger.error("Failed to connect to Hopsworks")
                return None
            
            logger.info("Training from Hopsworks Feature Store...")
            data = client.get_training_data()
            
            if data is None:
                logger.error("Failed to fetch training data from Hopsworks")
                return None
            
            X_train, X_test, y_train, y_test = data
            logger.info(f"Loaded {len(X_train)} train, {len(X_test)} test samples from Hopsworks")
            return self.train_from_dataframe(X_train, X_test, y_train, y_test)
            
        except Exception as e:
            logger.error(f"Training from Hopsworks failed: {str(e)}")
            return None

    def train_from_dataframe(self, X_train, X_test, y_train, y_test):
        try:
            self.feature_names = X_train.columns.tolist()
            
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_metrics = self._calculate_metrics(y_train, train_pred)
                test_metrics = self._calculate_metrics(y_test, test_pred)
                
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
            self.print_summary()
            return self
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return None

    def get_metrics_summary(self):
        """Get metrics summary for logging and display purposes"""
        metrics_data = {}
        for name, result in self.results.items():
            metrics_data[name] = {
                'train': result['train_metrics'],
                'test': result['test_metrics']
            }
        return metrics_data

    def print_summary(self):
        logger.info("Model training summary:")
        
        for name, result in self.results.items():
            logger.info(f"{name.upper()} - Training samples: {result['train_samples']}, Test samples: {result['test_samples']}")
            logger.info(f"Train R2: {result['train_metrics']['r2']:.3f}, Test R2: {result['test_metrics']['r2']:.3f}")
            logger.info(f"Train MAE: {result['train_metrics']['mae']:.3f}, Test MAE: {result['test_metrics']['mae']:.3f}")

    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 15):
        if model_name not in self.results:
            return None
        
        model = self.results[model_name]['model']
        importance = model.feature_importances_
        
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"Top {top_n} most important features:")
        logger.info(feature_imp.to_string(index=False))
        
        return feature_imp


def train_from_hopsworks():
    try:
        from src.hopsworks_client import HopsworksClient
        from src.training import AQIForecaster
        
        forecaster = AQIForecaster()
        
        client = HopsworksClient()
        if not client.connect():
            logger.error("Failed to connect to Hopsworks")
            return None
        
        logger.info("Training from Hopsworks Feature Store...")
        data = client.get_training_data()
        
        if data is None:
            logger.error("Failed to fetch training data from Hopsworks")
            return None
        
        forecaster = forecaster.train_from_hopsworks()
        
        if forecaster:
            feature_imp = forecaster.get_feature_importance('xgboost', top_n=20)
            
            logger.info("Saving models to Hopsworks Model Registry...")
            for name, result in forecaster.results.items():
                model = result['model']
                metrics = result['test_metrics']
                
                client.save_model(
                    model=model,
                    model_name=f"{config.HOPSWORKS_MODEL_NAME}_{name}",
                    metrics=metrics,
                    description=f"6-hour AQI forecasting model using {name}"
                )
        
        return forecaster
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None


class MultiHorizonForecaster:
    
    def __init__(self):
        self.models = {}  # Changed from horizon_models to just models
        self.horizons = config.FORECAST_HORIZONS
        self.feature_names = []
        
        # Initialize multi-output models (one per algorithm, outputs all 6 horizons)
        for name, params in config.MODELS_CONFIG.items():
            if name == 'random_forest':
                base_model = RandomForestRegressor(**params)
                self.models[name] = MultiOutputRegressor(base_model)
            elif name == 'ridge_regression':
                base_model = Ridge(**params)
                self.models[name] = MultiOutputRegressor(base_model)
    
    def prepare_multi_horizon_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Preparing multi-output data for all horizons...")
            
            # Create target variables for all horizons
            df_work = df.copy()
            
            # Create all horizon targets
            Y_columns = {}
            for horizon in self.horizons:
                Y_columns[horizon] = df_work['aqi'].shift(-horizon)
            
            # Combine into DataFrame (columns will be horizon indices)
            Y_df = pd.DataFrame(Y_columns)
            
            # Align with original dataframe
            df_work = pd.concat([df_work, Y_df], axis=1)
            
            # Remove rows with NaN targets (need to drop last max(horizons) rows)
            max_horizon = max(self.horizons)
            df_work = df_work.iloc[:-max_horizon] if max_horizon > 0 else df_work
            df_work = df_work.dropna(subset=list(self.horizons))
            
            if len(df_work) < 100:  # Minimum data requirement
                logger.warning(f"Insufficient data for multi-horizon: {len(df_work)} samples")
                return None, None
            
            # Prepare features
            # Exclude target, direct AQI, and non-lagged AQI features to prevent leakage
            numeric_features = df_work.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [
                c for c in numeric_features
                if c not in self.horizons and c != 'aqi' and c != 'aqi_6h_ahead' 
                and not (c.startswith('aqi_') and not c.startswith('aqi_lag_'))
            ]
            
            X = df_work[feature_columns]
            # Y should have columns in the same order as self.horizons
            Y = df_work[self.horizons]
            
            # Store feature names
            if not self.feature_names:
                self.feature_names = feature_columns
            
            logger.info(f"Prepared {len(X)} samples with {len(self.horizons)} horizon targets")
            logger.info(f"Feature shape: {X.shape}, Target shape: {Y.shape}")
            
            return X, Y
            
        except Exception as e:
            logger.error(f"Failed to prepare multi-horizon data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_multi_horizon_models(self, X: pd.DataFrame, Y: pd.DataFrame) -> Dict[str, Dict]:
        try:
            if X is None or Y is None or X.empty or Y.empty:
                logger.error("Invalid input data for multi-horizon training")
                return {}
            
            # Filter numeric columns only
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_features]
            
            # Chronological split
            split_idx = int(len(X_numeric) * config.TRAIN_TEST_SPLIT)
            X_train, X_test = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
            Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
            
            logger.info(f"Training multi-output models on {len(X_train)} train samples, {len(X_test)} test samples")
            
            results = {}
            
            # Train each multi-output model
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Training {model_name} multi-output model for all horizons...")
                    
                    # Train model (predicts all horizons at once)
                    model.fit(X_train, Y_train)
                    
                    # Make predictions (shape: n_samples x n_horizons)
                    Y_train_pred = model.predict(X_train)
                    Y_test_pred = model.predict(X_test)
                    
                    # Clip predictions to realistic AQI range (0-500)
                    Y_train_pred = np.clip(Y_train_pred, 0, 500)
                    Y_test_pred = np.clip(Y_test_pred, 0, 500)
                    
                    # Calculate metrics for each horizon
                    train_metrics_by_horizon = {}
                    test_metrics_by_horizon = {}
                    
                    for idx, horizon in enumerate(self.horizons):
                        y_train_true = Y_train.iloc[:, idx]
                        y_train_pred = Y_train_pred[:, idx]
                        y_test_true = Y_test.iloc[:, idx]
                        y_test_pred = Y_test_pred[:, idx]
                        
                        train_metrics = self._calculate_metrics(y_train_true, y_train_pred)
                        test_metrics = self._calculate_metrics(y_test_true, y_test_pred)
                        
                        train_metrics_by_horizon[horizon] = train_metrics
                        test_metrics_by_horizon[horizon] = test_metrics
                        
                        logger.info(f"  {horizon}h horizon - Test R²: {test_metrics['r2']:.3f}, MAE: {test_metrics['mae']:.3f}")
                    
                    results[model_name] = {
                        'model': model,
                        'train_metrics_by_horizon': train_metrics_by_horizon,
                        'test_metrics_by_horizon': test_metrics_by_horizon,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'horizons': self.horizons
                    }
                    
                    logger.info(f"{model_name} multi-output model trained successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name} multi-output model: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to train multi-horizon models: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics with robust handling of edge cases"""
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Handle NaN and infinite values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(valid_mask):
            # All values are invalid
            return {
                'mse': np.nan, 
                'rmse': np.nan, 
                'mae': np.nan, 
                'r2': -1.0, 
                'mape': np.nan
            }
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        # Handle constant target values (R² undefined)
        if len(np.unique(y_true_clean)) <= 1:
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = -1.0  # Return default for constant target
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-6))) * 100
        else:
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-6))) * 100
        
        # Handle potential NaN in R² (shouldn't happen with valid data, but be safe)
        if np.isnan(r2) or np.isinf(r2):
            r2 = -1.0
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
    
    def generate_multi_horizon_predictions(self, X_latest: pd.DataFrame, horizon_results: Dict[int, Dict[str, Dict]]) -> Dict[int, Dict[str, float]]:
        """Generate predictions for all horizons"""
        try:
            predictions = {}
            
            for horizon in self.horizons:
                if horizon not in horizon_results:
                    continue
                
                horizon_predictions = {}
                
                for model_name, model_data in horizon_results[horizon].items():
                    try:
                        model = model_data['model']
                        pred = model.predict(X_latest)[0]
                        horizon_predictions[model_name] = float(pred)
                        
                    except Exception as e:
                        logger.error(f"Failed to predict with {model_name} for {horizon}h: {str(e)}")
                        continue
                
                if horizon_predictions:
                    predictions[horizon] = horizon_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate multi-horizon predictions: {str(e)}")
            return {}
    
    def print_multi_horizon_summary(self, horizon_results: Dict[int, Dict[str, Dict]]):
        """Print summary of multi-horizon model performance"""
        logger.info("Multi-Horizon Forecasting Summary:")
        logger.info("=" * 50)
        
        for horizon in sorted(horizon_results.keys()):
            logger.info(f"\n{horizon}-Hour Horizon:")
            logger.info("-" * 20)
            
            for model_name, result in horizon_results[horizon].items():
                train_r2 = result['train_metrics']['r2']
                test_r2 = result['test_metrics']['r2']
                test_mae = result['test_metrics']['mae']
                
                logger.info(f"{model_name.upper()}:")
                logger.info(f"  Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
                logger.info(f"  Test MAE: {test_mae:.3f}")


def train_multi_horizon_from_hopsworks():
    """Train multi-horizon models from Hopsworks data"""
    try:
        logger.info("Starting multi-horizon forecasting training...")
        
        # Initialize multi-horizon forecaster
        forecaster = MultiHorizonForecaster()
        
        # Connect to Hopsworks
        client = HopsworksClient()
        if not client.connect():
            logger.error("Failed to connect to Hopsworks")
            return None
        
        # Get feature data
        logger.info("Loading data from Hopsworks...")
        df = client.get_feature_data()
        
        if df is None or df.empty:
            logger.error("No data available from Hopsworks")
            return None
        
        logger.info(f"Loaded {len(df)} records for multi-horizon training")
        
        # Prepare multi-horizon data
        horizon_data = forecaster.prepare_multi_horizon_data(df)
        
        if not horizon_data:
            logger.error("Failed to prepare multi-horizon data")
            return None
        
        # Train models for all horizons
        horizon_results = forecaster.train_multi_horizon_models(horizon_data)
        
        if not horizon_results:
            logger.error("Failed to train multi-horizon models")
            return None
        
        # Print summary
        forecaster.print_multi_horizon_summary(horizon_results)
        
        logger.info("Multi-horizon forecasting training completed successfully")
        return forecaster
        
    except Exception as e:
        logger.error(f"Multi-horizon training failed: {str(e)}")
        return None


if __name__ == "__main__":
    train_from_hopsworks()