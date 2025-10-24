import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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
    """
    Multi-horizon forecasting for 3-day ahead AQI predictions
    Supports multiple time horizons: 1h, 6h, 12h, 24h, 48h, 72h
    """
    
    def __init__(self):
        self.horizon_models = {}
        self.horizons = config.FORECAST_HORIZONS
        self.feature_names = []
        
        # Initialize models for each horizon
        for horizon in self.horizons:
            self.horizon_models[horizon] = {}
            for name, params in config.MODELS_CONFIG.items():
                if name == 'random_forest':
                    self.horizon_models[horizon][name] = RandomForestRegressor(**params)
                elif name == 'ridge_regression':
                    self.horizon_models[horizon][name] = Ridge(**params)
    
    def prepare_multi_horizon_data(self, df: pd.DataFrame) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for multiple forecasting horizons"""
        try:
            horizon_data = {}
            
            for horizon in self.horizons:
                logger.info(f"Preparing data for {horizon}-hour horizon...")
                
                # Create target variable for this horizon
                df_horizon = df.copy()
                df_horizon['target'] = df_horizon['aqi'].shift(-horizon)
                
                # Remove rows with NaN targets (last 'horizon' rows)
                df_horizon = df_horizon.dropna(subset=['target'])
                
                if len(df_horizon) < 100:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {horizon}-hour horizon: {len(df_horizon)} samples")
                    continue
                
                # Prepare features and target
                feature_columns = [col for col in df_horizon.columns if col not in ['target', 'aqi']]
                X = df_horizon[feature_columns]
                y = df_horizon['target']
                
                # Store feature names (same for all horizons)
                if not self.feature_names:
                    self.feature_names = feature_columns
                
                horizon_data[horizon] = (X, y)
                logger.info(f"Prepared {len(X)} samples for {horizon}-hour horizon")
            
            return horizon_data
            
        except Exception as e:
            logger.error(f"Failed to prepare multi-horizon data: {str(e)}")
            return {}
    
    def train_multi_horizon_models(self, horizon_data: Dict[int, Tuple[pd.DataFrame, pd.Series]]) -> Dict[int, Dict[str, Dict]]:
        """Train models for all horizons"""
        try:
            results = {}
            
            for horizon, (X, y) in horizon_data.items():
                logger.info(f"Training models for {horizon}-hour horizon...")
                
                # Filter numeric columns only
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                X_numeric = X[numeric_features]
                
                # Split data
                split_idx = int(len(X_numeric) * config.TRAIN_TEST_SPLIT)
                X_train, X_test = X_numeric.iloc[:split_idx], X_numeric.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                horizon_results = {}
                
                # Train each model for this horizon
                for model_name, model in self.horizon_models[horizon].items():
                    try:
                        logger.info(f"Training {model_name} for {horizon}-hour horizon...")
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        train_metrics = self._calculate_metrics(y_train, y_train_pred)
                        test_metrics = self._calculate_metrics(y_test, y_test_pred)
                        
                        horizon_results[model_name] = {
                            'model': model,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'train_samples': len(X_train),
                            'test_samples': len(X_test)
                        }
                        
                        logger.info(f"{model_name} for {horizon}h - Test R²: {test_metrics['r2']:.3f}, MAE: {test_metrics['mae']:.3f}")
                        
                    except Exception as e:
                        logger.error(f"Failed to train {model_name} for {horizon}h: {str(e)}")
                        continue
                
                results[horizon] = horizon_results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to train multi-horizon models: {str(e)}")
            return {}
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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