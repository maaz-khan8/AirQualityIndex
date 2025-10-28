import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, List
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


class HopsworksClient:
    def __init__(self):
        self.project = None
        self.fs = None
        self.fv = None

    def connect(self):
        try:
            self.project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
            self.fs = self.project.get_feature_store()
            logger.info("Connected to Hopsworks")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False

    def create_feature_group(self, df: pd.DataFrame, name: Optional[str] = None, 
                           version: Optional[int] = None, description: str = None,
                           primary_key: List[str] = None, event_time: str = "timestamp"):
        
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Call connect() first.")
            return None
        
        name = name or config.HOPSWORKS_FEATURE_GROUP_NAME
        version = version or config.HOPSWORKS_FEATURE_GROUP_VERSION
        primary_key = primary_key or ["timestamp", "city"]
        
        try:
            logger.info(f"Creating/getting feature group: {name} v{version}")
            
            fg = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=False
            )
            
            logger.info(f"Feature group '{name}' ready")
            
            logger.info(f"Inserting {len(df)} records into feature group...")
            fg.insert(df, write_options={"wait_for_job": True})
            
            logger.info(f"Successfully inserted {len(df)} records")
            return fg
            
        except Exception as e:
            logger.error(f"Failed to create/update feature group: {str(e)}")
            return None

    def create_feature_view(self, name: Optional[str] = None,
                          version: Optional[int] = None,
                          description: str = "Feature view for 6-hour AQI forecasting",
                          labels: List[str] = None):
        
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Call connect() first.")
            return None
        
        name = name or config.HOPSWORKS_FEATURE_VIEW_NAME
        version = version or config.HOPSWORKS_FEATURE_VIEW_VERSION
        labels = labels or ["aqi_6h_ahead"]
        
        try:
            fg_name = config.HOPSWORKS_FEATURE_GROUP_NAME
            fg_version = config.HOPSWORKS_FEATURE_GROUP_VERSION
            
            fg = self.fs.get_feature_group(name=fg_name, version=fg_version)
            
            fv = self.fs.get_or_create_feature_view(
                name=name,
                version=version,
                description=description,
                labels=labels,
                query=fg.select_all()
            )
            
            self.fv = fv
            logger.info(f"Feature view created: {name}")
            return fv
            
        except Exception as e:
            logger.error(f"Failed to create feature view: {str(e)}")
            return None

    def get_training_data(self, feature_view_name: Optional[str] = None,
                         feature_view_version: Optional[int] = None,
                         train_test_split: float = None):
        
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Call connect() first.")
            return None
        
        feature_view_name = feature_view_name or config.HOPSWORKS_FEATURE_VIEW_NAME
        feature_view_version = feature_view_version or config.HOPSWORKS_FEATURE_VIEW_VERSION
        train_test_split = train_test_split or config.TRAIN_TEST_SPLIT
        
        try:
            logger.info(f"Reading data directly from feature group...")
            
            fg = self.fs.get_feature_group(
                name=config.HOPSWORKS_FEATURE_GROUP_NAME,
                version=config.HOPSWORKS_FEATURE_GROUP_VERSION
            )
            
            logger.info("Reading all data from feature group...")
            df = fg.read()
            
            logger.info(f"Loaded {len(df)} records from feature group")
            
            target_col = "aqi_6h_ahead"
            drop_cols = [target_col, "timestamp", "city"]
            drop_cols = [col for col in drop_cols if col in df.columns]
            
            X = df.drop(columns=drop_cols)
            y = df[target_col]
            
            from sklearn.model_selection import train_test_split as sklearn_train_test_split
            X_train, X_test, y_train, y_test = sklearn_train_test_split(
                X, y, test_size=1 - train_test_split, random_state=42
            )
            
            logger.info(f"Training data prepared: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to get training data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_model(self, model, model_name: Optional[str] = None,
                  metrics: dict = None, description: str = "6-hour AQI forecasting model"):
        
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Call connect() first.")
            return None
        
        model_name = model_name or config.HOPSWORKS_MODEL_NAME
        
        try:
            mr = self.project.get_model_registry()
            
            model_obj = mr.python.create_model(
                name=model_name,
                description=description
            )
            
            model_obj.save(model)
            
            # Some Hopsworks model objects don't expose save_metric; avoid hard failure
            try:
                if metrics and hasattr(model_obj, "save_metric"):
                    for metric_name, metric_value in metrics.items():
                        model_obj.save_metric(metric_name, metric_value)
            except Exception:
                logger.warning("Model metrics not saved (save_metric not available)")
            
            logger.info(f"Model saved to registry: {model_name}")
            return model_obj
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return None

    def get_feature_data(self, limit: int = 1000):
        if self.fs is None:
            logger.error("Not connected to Hopsworks. Call connect() first.")
            return None
        
        try:
            fv = self.fs.get_feature_view(
                name=config.HOPSWORKS_FEATURE_VIEW_NAME,
                version=config.HOPSWORKS_FEATURE_VIEW_VERSION
            )
            
            if fv is None:
                logger.error("Failed to get feature view")
                return None
            
            batch_result = fv.get_batch_data()
            if isinstance(batch_result, tuple):
                data = batch_result[0]
            else:
                data = batch_result
            
            if data is None or len(data) == 0:
                logger.warning("No data found in feature view")
                return None
            
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp', ascending=False)
            
            data = data.head(limit)
            
            logger.info(f"Retrieved {len(data)} feature records for dashboard")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get feature data: {str(e)}")
            return None

    def load_models(self):
        """Load trained models for dashboard usage.
        Primary source: local 'models' directory where pipeline saves .pkl files.
        If not found, attempts to fetch latest from Hopsworks Model Registry.
        Returns a dict of {model_name: loaded_model}.
        """
        try:
            import os
            import glob
            import joblib
            models = {}

            # 1) Try local models first
            local_dir = os.path.join(os.getcwd(), "models")
            if os.path.isdir(local_dir):
                for path in glob.glob(os.path.join(local_dir, "*_model.pkl")):
                    name = os.path.basename(path).replace("_model.pkl", "")
                    try:
                        models[name] = joblib.load(path)
                    except Exception:
                        continue

            # 2) If none loaded and connected, try Hopsworks registry
            if not models and self.project is not None:
                try:
                    mr = self.project.get_model_registry()
                    # Expect names saved by pipeline
                    candidate_names = [
                        f"{config.HOPSWORKS_MODEL_NAME}_random_forest",
                        f"{config.HOPSWORKS_MODEL_NAME}_ridge_regression",
                    ]
                    for mname in candidate_names:
                        try:
                            model = mr.get_model(name=mname, version=None)  # latest
                            model_dir = model.download()
                            pkl_paths = glob.glob(os.path.join(model_dir, "**", "*.pkl"), recursive=True)
                            if pkl_paths:
                                models[mname] = joblib.load(pkl_paths[0])
                        except Exception:
                            continue
                except Exception:
                    pass

            return models
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return {}