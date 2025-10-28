"""
Model Registry Metrics Loader
Loads and displays model metrics from sidecar JSON files
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelMetricsLoader:
    """
    Loads model metrics from sidecar JSON files and provides utilities for display
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
    
    def load_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Load metrics for all models in the models directory
        
        Returns:
            Dictionary of model_name -> metrics_data
        """
        metrics_data = {}
        
        try:
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory {self.models_dir} does not exist")
                return metrics_data
            
            # Find all metrics JSON files
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_metrics.json'):
                    model_name = filename.replace('_metrics.json', '')
                    metrics_path = os.path.join(self.models_dir, filename)
                    
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics_info = json.load(f)
                        
                        metrics_data[model_name] = metrics_info
                        logger.info(f"Loaded metrics for {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load metrics for {model_name}: {str(e)}")
            
            logger.info(f"Loaded metrics for {len(metrics_data)} models")
            return metrics_data
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {str(e)}")
            return metrics_data
    
    def load_model_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metrics for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metrics data dictionary or None if not found
        """
        try:
            metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")
            
            if not os.path.exists(metrics_path):
                logger.warning(f"Metrics file not found for {model_name}")
                return None
            
            with open(metrics_path, 'r') as f:
                metrics_info = json.load(f)
            
            return metrics_info
            
        except Exception as e:
            logger.error(f"Failed to load metrics for {model_name}: {str(e)}")
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all model metrics
        
        Returns:
            Summary dictionary with model comparisons
        """
        try:
            all_metrics = self.load_all_metrics()
            
            if not all_metrics:
                return {
                    'total_models': 0,
                    'models': [],
                    'best_models': {},
                    'summary': 'No model metrics found'
                }
            
            # Extract key metrics for comparison
            model_summaries = []
            best_models = {
                'mae': {'model': None, 'value': float('inf')},
                'r2': {'model': None, 'value': float('-inf')},
                'rmse': {'model': None, 'value': float('inf')}
            }
            
            for model_name, metrics_info in all_metrics.items():
                metrics = metrics_info.get('metrics', {})
                
                summary = {
                    'model_name': model_name,
                    'algorithm': metrics_info.get('algorithm', 'unknown'),
                    'horizon': metrics_info.get('horizon', 'N/A'),
                    'training_timestamp': metrics_info.get('training_timestamp', 'unknown'),
                    'mae': metrics.get('mae', 0),
                    'r2': metrics.get('r2', 0),
                    'rmse': metrics.get('rmse', 0),
                    'version': metrics_info.get('version', '1.0')
                }
                
                model_summaries.append(summary)
                
                # Track best models
                if metrics.get('mae', float('inf')) < best_models['mae']['value']:
                    best_models['mae'] = {'model': model_name, 'value': metrics.get('mae', 0)}
                
                if metrics.get('r2', float('-inf')) > best_models['r2']['value']:
                    best_models['r2'] = {'model': model_name, 'value': metrics.get('r2', 0)}
                
                if metrics.get('rmse', float('inf')) < best_models['rmse']['value']:
                    best_models['rmse'] = {'model': model_name, 'value': metrics.get('rmse', 0)}
            
            # Sort models by R² score (descending)
            model_summaries.sort(key=lambda x: x['r2'], reverse=True)
            
            return {
                'total_models': len(all_metrics),
                'models': model_summaries,
                'best_models': best_models,
                'summary': f"Loaded metrics for {len(all_metrics)} models"
            }
            
        except Exception as e:
            logger.error(f"Failed to create metrics summary: {str(e)}")
            return {'error': str(e)}
    
    def get_model_comparison_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get model metrics as a pandas DataFrame for easy comparison
        
        Returns:
            DataFrame with model metrics or None if no data
        """
        try:
            summary = self.get_metrics_summary()
            
            if not summary.get('models'):
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(summary['models'])
            
            # Format numeric columns
            numeric_cols = ['mae', 'r2', 'rmse']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Round numeric values
            df[numeric_cols] = df[numeric_cols].round(4)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to create comparison DataFrame: {str(e)}")
            return None
    
    def get_horizon_performance(self) -> Dict[int, Dict[str, Any]]:
        """
        Get performance metrics grouped by horizon
        
        Returns:
            Dictionary of horizon -> performance metrics
        """
        try:
            all_metrics = self.load_all_metrics()
            horizon_performance = {}
            
            for model_name, metrics_info in all_metrics.items():
                horizon = metrics_info.get('horizon')
                if horizon is None:
                    continue
                
                if horizon not in horizon_performance:
                    horizon_performance[horizon] = {
                        'models': [],
                        'best_mae': float('inf'),
                        'best_r2': float('-inf'),
                        'avg_mae': 0,
                        'avg_r2': 0
                    }
                
                metrics = metrics_info.get('metrics', {})
                model_perf = {
                    'model_name': model_name,
                    'algorithm': metrics_info.get('algorithm', 'unknown'),
                    'mae': metrics.get('mae', 0),
                    'r2': metrics.get('r2', 0),
                    'rmse': metrics.get('rmse', 0)
                }
                
                horizon_performance[horizon]['models'].append(model_perf)
                
                # Track best performance
                if metrics.get('mae', float('inf')) < horizon_performance[horizon]['best_mae']:
                    horizon_performance[horizon]['best_mae'] = metrics.get('mae', 0)
                
                if metrics.get('r2', float('-inf')) > horizon_performance[horizon]['best_r2']:
                    horizon_performance[horizon]['best_r2'] = metrics.get('r2', 0)
            
            # Calculate averages
            for horizon, data in horizon_performance.items():
                if data['models']:
                    data['avg_mae'] = sum(m['mae'] for m in data['models']) / len(data['models'])
                    data['avg_r2'] = sum(m['r2'] for m in data['models']) / len(data['models'])
            
            return horizon_performance
            
        except Exception as e:
            logger.error(f"Failed to get horizon performance: {str(e)}")
            return {}
    
    def get_latest_model_versions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest version of each model type
        
        Returns:
            Dictionary of model_type -> latest_metrics
        """
        try:
            all_metrics = self.load_all_metrics()
            latest_models = {}
            
            for model_name, metrics_info in all_metrics.items():
                # Extract model type (algorithm)
                algorithm = metrics_info.get('algorithm', 'unknown')
                training_time = metrics_info.get('training_timestamp', '')
                
                if algorithm not in latest_models:
                    latest_models[algorithm] = {
                        'model_name': model_name,
                        'metrics_info': metrics_info,
                        'training_timestamp': training_time
                    }
                else:
                    # Compare timestamps to find latest
                    current_time = latest_models[algorithm]['training_timestamp']
                    if training_time > current_time:
                        latest_models[algorithm] = {
                            'model_name': model_name,
                            'metrics_info': metrics_info,
                            'training_timestamp': training_time
                        }
            
            return latest_models
            
        except Exception as e:
            logger.error(f"Failed to get latest model versions: {str(e)}")
            return {}


def load_model_metrics_summary() -> Dict[str, Any]:
    """
    Convenience function to load model metrics summary
    
    Returns:
        Model metrics summary dictionary
    """
    try:
        loader = ModelMetricsLoader()
        return loader.get_metrics_summary()
    except Exception as e:
        logger.error(f"Failed to load metrics summary: {str(e)}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Test metrics loader
    loader = ModelMetricsLoader()
    
    # Load all metrics
    all_metrics = loader.load_all_metrics()
    print(f"Loaded metrics for {len(all_metrics)} models")
    
    # Get summary
    summary = loader.get_metrics_summary()
    print(f"Summary: {summary['summary']}")
    
    # Get comparison DataFrame
    df = loader.get_model_comparison_dataframe()
    if df is not None:
        print(f"Comparison DataFrame shape: {df.shape}")
        print(df.head())
    
    print("✅ Metrics loader test completed!")
