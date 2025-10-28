"""
Model Interpretability using SHAP (SHapley Additive exPlanations)
Provides explanations for model predictions and feature importance
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based model interpretability analyzer
    """
    
    def __init__(self):
        self.shap_available = SHAP_AVAILABLE
        if not self.shap_available:
            logger.error("SHAP not available. Please install: pip install shap")
        
        # Create artifacts directory
        self.artifacts_dir = "artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def analyze_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                     model_name: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform SHAP analysis on a trained model
        
        Args:
            model: Trained model object
            X_train: Training features
            X_test: Test features  
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        if not self.shap_available:
            logger.error("SHAP not available")
            return {}
        
        try:
            logger.info(f"Starting SHAP analysis for {model_name}")
            
            # Use feature names from DataFrame if not provided
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            
            # Select appropriate SHAP explainer based on model type
            explainer = self._get_explainer(model, X_train)
            
            if explainer is None:
                logger.error(f"Could not create SHAP explainer for {model_name}")
                return {}
            
            # Calculate SHAP values for test set (sample for performance)
            test_sample_size = min(100, len(X_test))
            X_test_sample = X_test.sample(n=test_sample_size, random_state=42)
            
            logger.info(f"Calculating SHAP values for {test_sample_size} test samples")
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-class, take first class
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Get top features
            top_features = feature_importance_df.head(10)
            
            # Save artifacts
            artifacts = self._save_shap_artifacts(
                model_name, shap_values, X_test_sample, feature_names,
                feature_importance_df, top_features
            )
            
            # Create summary
            summary = {
                'model_name': model_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'test_samples_analyzed': test_sample_size,
                'total_features': len(feature_names),
                'top_features': top_features.to_dict('records'),
                'artifacts': artifacts
            }
            
            logger.info(f"SHAP analysis completed for {model_name}")
            logger.info(f"Top 5 features: {top_features.head()['feature'].tolist()}")
            
            return summary
            
        except Exception as e:
            logger.error(f"SHAP analysis failed for {model_name}: {str(e)}")
            return {}
    
    def _get_explainer(self, model, X_train: pd.DataFrame):
        """Get appropriate SHAP explainer for model type"""
        try:
            model_type = type(model).__name__.lower()
            
            if 'randomforest' in model_type or 'tree' in model_type:
                logger.info("Using TreeExplainer for tree-based model")
                return shap.TreeExplainer(model)
            
            elif 'linear' in model_type or 'ridge' in model_type or 'lasso' in model_type:
                logger.info("Using LinearExplainer for linear model")
                return shap.LinearExplainer(model, X_train)
            
            else:
                # Fallback to KernelExplainer (slower but more general)
                logger.info("Using KernelExplainer as fallback")
                # Use a subset of training data for background
                background_size = min(50, len(X_train))
                background = X_train.sample(n=background_size, random_state=42)
                return shap.KernelExplainer(model.predict, background)
                
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {str(e)}")
            return None
    
    def _save_shap_artifacts(self, model_name: str, shap_values: np.ndarray, 
                           X_test_sample: pd.DataFrame, feature_names: List[str],
                           feature_importance_df: pd.DataFrame, 
                           top_features: pd.DataFrame) -> Dict[str, str]:
        """Save SHAP analysis artifacts"""
        artifacts = {}
        
        try:
            # Save top features as JSON
            top_features_file = f"{self.artifacts_dir}/shap_top_features_{model_name}.json"
            top_features.to_json(top_features_file, orient='records', indent=2)
            artifacts['top_features_json'] = top_features_file
            
            # Save full feature importance
            importance_file = f"{self.artifacts_dir}/shap_feature_importance_{model_name}.json"
            feature_importance_df.to_json(importance_file, orient='records', indent=2)
            artifacts['feature_importance_json'] = importance_file
            
            # Create and save plots
            plot_files = self._create_shap_plots(
                model_name, shap_values, X_test_sample, feature_names, top_features
            )
            artifacts.update(plot_files)
            
            logger.info(f"SHAP artifacts saved for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save SHAP artifacts: {str(e)}")
        
        return artifacts
    
    def _create_shap_plots(self, model_name: str, shap_values: np.ndarray,
                          X_test_sample: pd.DataFrame, feature_names: List[str],
                          top_features: pd.DataFrame) -> Dict[str, str]:
        """Create SHAP visualization plots"""
        plot_files = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Feature importance bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = top_features.head(10)
            ax.barh(range(len(top_10)), top_10['importance'])
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['feature'])
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(f'SHAP Feature Importance - {model_name}')
            ax.invert_yaxis()
            
            importance_plot = f"{self.artifacts_dir}/shap_importance_{model_name}.png"
            plt.tight_layout()
            plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['importance_plot'] = importance_plot
            
            # 2. SHAP summary plot (if we have reasonable number of features)
            if len(feature_names) <= 20:  # Only for manageable number of features
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create summary plot data
                shap_df = pd.DataFrame(shap_values, columns=feature_names)
                
                # Plot top 10 features
                top_feature_names = top_features.head(10)['feature'].tolist()
                shap_top = shap_df[top_feature_names]
                
                # Create beeswarm-style plot
                for i, feature in enumerate(top_feature_names):
                    values = shap_top[feature]
                    colors = ['red' if v > 0 else 'blue' for v in values]
                    ax.scatter(values, [i] * len(values), c=colors, alpha=0.6, s=20)
                
                ax.set_yticks(range(len(top_feature_names)))
                ax.set_yticklabels(top_feature_names)
                ax.set_xlabel('SHAP value')
                ax.set_title(f'SHAP Values Distribution - {model_name}')
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                summary_plot = f"{self.artifacts_dir}/shap_summary_{model_name}.png"
                plt.tight_layout()
                plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['summary_plot'] = summary_plot
            
            logger.info(f"SHAP plots created for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to create SHAP plots: {str(e)}")
        
        return plot_files
    
    def load_analysis_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load previously saved SHAP analysis results"""
        try:
            top_features_file = f"{self.artifacts_dir}/shap_top_features_{model_name}.json"
            
            if not os.path.exists(top_features_file):
                logger.warning(f"No SHAP analysis found for {model_name}")
                return None
            
            with open(top_features_file, 'r') as f:
                top_features = json.load(f)
            
            return {
                'model_name': model_name,
                'top_features': top_features,
                'artifacts_dir': self.artifacts_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to load SHAP analysis for {model_name}: {str(e)}")
            return None
    
    def get_model_explanation(self, model, X_sample: pd.DataFrame, 
                            feature_names: List[str] = None) -> Optional[Dict[str, Any]]:
        """Get SHAP explanation for a single prediction"""
        if not self.shap_available:
            return None
        
        try:
            explainer = self._get_explainer(model, X_sample)
            if explainer is None:
                return None
            
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if feature_names is None:
                feature_names = X_sample.columns.tolist()
            
            # Create explanation
            explanation = {
                'prediction': model.predict(X_sample)[0],
                'feature_contributions': dict(zip(feature_names, shap_values[0])),
                'feature_values': dict(zip(feature_names, X_sample.iloc[0].values))
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate model explanation: {str(e)}")
            return None


def run_shap_analysis(models: Dict[str, Any], X_train: pd.DataFrame, 
                     X_test: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run SHAP analysis on multiple models
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        
    Returns:
        Dictionary of SHAP analysis results for each model
    """
    analyzer = SHAPAnalyzer()
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Running SHAP analysis for {model_name}")
        result = analyzer.analyze_model(model, X_train, X_test, model_name)
        if result:
            results[model_name] = result
    
    return results


if __name__ == "__main__":
    # Test SHAP availability
    if SHAP_AVAILABLE:
        print("✅ SHAP is available and ready to use")
    else:
        print("❌ SHAP not available. Install with: pip install shap")
