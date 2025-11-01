"""
Exploratory Data Analysis (EDA) Snapshot Module
Generates lightweight EDA reports with distributions, seasonality, and correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class EDASnapshot:
    """
    Generates exploratory data analysis snapshots for air quality data
    """
    
    def __init__(self):
        # Create reports directory
        self.reports_dir = "reports"
        self.latest_dir = os.path.join(self.reports_dir, "latest")
        os.makedirs(self.latest_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_eda_report(self, df: pd.DataFrame, data_source: str = "unknown") -> Dict[str, Any]:
        """
        Generate comprehensive EDA report
        
        Args:
            df: DataFrame to analyze
            data_source: Source identifier for the data
            
        Returns:
            EDA report results dictionary
        """
        try:
            logger.info(f"Generating EDA report for {data_source}")
            
            if df.empty:
                logger.warning("Empty dataset provided for EDA")
                return {'status': 'ERROR', 'error': 'Empty dataset'}
            
            # Initialize report
            report = {
                'data_source': data_source,
                'generation_timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'total_columns': len(df.columns),
                'analysis_period': self._get_analysis_period(df),
                'status': 'SUCCESS',
                'artifacts': {}
            }
            
            # 1. Data Overview
            logger.info("Generating data overview...")
            overview = self._generate_data_overview(df)
            report['overview'] = overview
            
            # 2. Distribution Analysis
            logger.info("Analyzing distributions...")
            distributions = self._analyze_distributions(df)
            report['distributions'] = distributions
            
            # 3. Time Series Analysis
            logger.info("Analyzing time series patterns...")
            time_series = self._analyze_time_series(df)
            report['time_series'] = time_series
            
            # 4. Correlation Analysis
            logger.info("Analyzing correlations...")
            correlations = self._analyze_correlations(df)
            report['correlations'] = correlations
            
            # 5. Missing Data Analysis
            logger.info("Analyzing missing data...")
            missing_data = self._analyze_missing_data(df)
            report['missing_data'] = missing_data
            
            # 6. Generate Visualizations
            logger.info("Generating visualizations...")
            artifacts = self._generate_visualizations(df, report)
            report['artifacts'] = artifacts
            
            # Save report metadata
            self._save_report_metadata(report)
            
            logger.info(f"EDA report generated successfully with {len(artifacts)} visualizations")
            return report
            
        except Exception as e:
            logger.error(f"EDA report generation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _get_analysis_period(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get analysis period from data"""
        try:
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                return {
                    'start': timestamps.min().isoformat(),
                    'end': timestamps.max().isoformat(),
                    'duration_days': (timestamps.max() - timestamps.min()).days
                }
            else:
                return {'start': 'unknown', 'end': 'unknown', 'duration_days': 0}
        except Exception as e:
            logger.error(f"Failed to get analysis period: {str(e)}")
            return {'start': 'error', 'end': 'error', 'duration_days': 0}
    
    def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data overview"""
        try:
            overview = {
                'shape': list(df.shape),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Basic statistics for numeric columns
            if overview['numeric_columns']:
                basic_stats = df[overview['numeric_columns']].describe()
                overview['basic_stats'] = {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in basic_stats.to_dict().items()}
            
            return overview
            
        except Exception as e:
            logger.error(f"Data overview generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions"""
        try:
            distributions = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col in df.columns and not df[col].isnull().all():
                    series = df[col].dropna()
                    
                    distributions[col] = {
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'skewness': float(stats.skew(series)) if len(series) > 1 else 0.0,
                        'kurtosis': float(stats.kurtosis(series)) if len(series) > 1 else 0.0,
                        'percentiles': {
                            '25th': float(series.quantile(0.25)),
                            '75th': float(series.quantile(0.75)),
                            '90th': float(series.quantile(0.90)),
                            '95th': float(series.quantile(0.95))
                        }
                    }
            
            return distributions
            
        except Exception as e:
            logger.error(f"Distribution analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series patterns"""
        try:
            time_series = {}
            
            if 'timestamp' not in df.columns:
                return {'error': 'No timestamp column found'}
            
            # Convert timestamp to datetime
            df_ts = df.copy()
            df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
            df_ts = df_ts.sort_values('timestamp')
            
            # Analyze AQI trends
            if 'aqi' in df_ts.columns:
                aqi_series = df_ts.set_index('timestamp')['aqi']
                
                # Daily aggregation
                daily_aqi = aqi_series.resample('D').mean()
                
                time_series['aqi'] = {
                    'daily_mean': float(daily_aqi.mean()),
                    'daily_std': float(daily_aqi.std()),
                    'trend_slope': float(np.polyfit(range(len(daily_aqi)), daily_aqi.values, 1)[0]),
                    'seasonality_detected': self._detect_seasonality(daily_aqi),
                    'outliers_count': len(self._detect_outliers(aqi_series))
                }
            
            # Analyze pollutant trends
            pollutant_cols = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
            for col in pollutant_cols:
                if col in df_ts.columns:
                    series = df_ts.set_index('timestamp')[col].dropna()
                    if not series.empty:
                        daily_series = series.resample('D').mean()
                        
                        time_series[col] = {
                            'daily_mean': float(daily_series.mean()),
                            'daily_std': float(daily_series.std()),
                            'trend_slope': float(np.polyfit(range(len(daily_series)), daily_series.values, 1)[0]),
                            'outliers_count': len(self._detect_outliers(series))
                        }
            
            return time_series
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Detect seasonality in time series"""
        try:
            if len(series) < 30:  # Need at least 30 days
                return False
            
            # Simple seasonality detection using autocorrelation
            autocorr = series.autocorr(lag=7)  # Weekly seasonality
            return abs(autocorr) > 0.3
            
        except Exception as e:
            logger.error(f"Seasonality detection failed: {str(e)}")
            return False
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers.index.tolist()
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {str(e)}")
            return []
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables"""
        try:
            correlations = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                # Find strong correlations (>0.7 or <-0.7)
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': float(corr_value)
                            })
                
                correlations['strong_correlations'] = strong_correlations
                correlations['correlation_matrix'] = corr_matrix.to_dict()
                
                # AQI correlations
                if 'aqi' in numeric_cols:
                    aqi_correlations = corr_matrix['aqi'].drop('aqi').sort_values(key=abs, ascending=False)
                    correlations['aqi_correlations'] = aqi_correlations.to_dict()
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        try:
            missing_data = {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_missing': {},
                'missing_patterns': {}
            }
            
            # Missing data by column
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_data['columns_with_missing'][col] = {
                        'count': int(missing_count),
                        'percentage': float((missing_count / len(df)) * 100)
                    }
            
            return missing_data
            
        except Exception as e:
            logger.error(f"Missing data analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_visualizations(self, df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization plots"""
        try:
            artifacts = {}
            
            # 1. Distribution plots
            dist_plot = self._create_distribution_plots(df)
            if dist_plot:
                artifacts['distribution_plot'] = dist_plot
            
            # 2. Time series plots
            ts_plot = self._create_time_series_plots(df)
            if ts_plot:
                artifacts['time_series_plot'] = ts_plot
            
            # 3. Correlation heatmap
            corr_plot = self._create_correlation_heatmap(df)
            if corr_plot:
                artifacts['correlation_plot'] = corr_plot
            
            # 4. Missing data plot
            missing_plot = self._create_missing_data_plot(df)
            if missing_plot:
                artifacts['missing_data_plot'] = missing_plot
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return {}
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> Optional[str]:
        """Create distribution plots"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None
            
            # Select top 6 numeric columns for plotting
            plot_cols = numeric_cols[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(plot_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(plot_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.latest_dir, 'distributions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Distribution plot creation failed: {str(e)}")
            return None
    
    def _create_time_series_plots(self, df: pd.DataFrame) -> Optional[str]:
        """Create time series plots"""
        try:
            if 'timestamp' not in df.columns:
                return None
            
            df_ts = df.copy()
            df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
            df_ts = df_ts.sort_values('timestamp')
            
            # Plot AQI and key pollutants
            plot_cols = ['aqi', 'pm2_5', 'pm10', 'o3']
            available_cols = [col for col in plot_cols if col in df_ts.columns]
            
            if not available_cols:
                return None
            
            fig, axes = plt.subplots(len(available_cols), 1, figsize=(15, 4*len(available_cols)))
            if len(available_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(available_cols):
                axes[i].plot(df_ts['timestamp'], df_ts[col], alpha=0.7)
                axes[i].set_title(f'{col} Time Series')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.latest_dir, 'time_series.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Time series plot creation failed: {str(e)}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> Optional[str]:
        """Create correlation heatmap"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return None
            
            corr_matrix = df[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            plot_path = os.path.join(self.latest_dir, 'correlation_heatmap.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Correlation heatmap creation failed: {str(e)}")
            return None
    
    def _create_missing_data_plot(self, df: pd.DataFrame) -> Optional[str]:
        """Create missing data visualization"""
        try:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if missing_data.empty:
                return None
            
            plt.figure(figsize=(10, 6))
            missing_data.plot(kind='bar')
            plt.title('Missing Data by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(self.latest_dir, 'missing_data.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Missing data plot creation failed: {str(e)}")
            return None
    
    
    def _save_report_metadata(self, report: Dict[str, Any]):
        """Save report metadata"""
        try:
            metadata_path = os.path.join(self.latest_dir, 'eda_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"EDA metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save EDA metadata: {str(e)}")
    
    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest EDA report metadata"""
        try:
            metadata_path = os.path.join(self.latest_dir, 'eda_metadata.json')
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load latest EDA report: {str(e)}")
            return None


def generate_eda_snapshot(df: pd.DataFrame, data_source: str = "unknown") -> Dict[str, Any]:
    """
    Convenience function to generate EDA snapshot
    
    Args:
        df: DataFrame to analyze
        data_source: Source identifier
        
    Returns:
        EDA report dictionary
    """
    try:
        eda = EDASnapshot()
        return eda.generate_eda_report(df, data_source)
    except Exception as e:
        logger.error(f"EDA snapshot generation failed: {str(e)}")
        return {'status': 'ERROR', 'error': str(e)}


if __name__ == "__main__":
    # Test EDA snapshot
    eda = EDASnapshot()
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-10-01', periods=100, freq='H'),
        'city': ['London'] * 100,
        'latitude': [51.5074] * 100,
        'longitude': [-0.1278] * 100,
        'aqi': np.random.normal(50, 20, 100),
        'pm2_5': np.random.normal(25, 10, 100),
        'pm10': np.random.normal(40, 15, 100),
        'o3': np.random.normal(60, 20, 100),
        'no2': np.random.normal(30, 10, 100),
        'so2': np.random.normal(10, 5, 100),
        'co': np.random.normal(1, 0.5, 100),
        'temperature_2m': np.random.normal(15, 5, 100),
        'relative_humidity_2m': np.random.normal(70, 10, 100),
        'pressure_msl': np.random.normal(1013, 10, 100),
        'wind_speed_10m': np.random.normal(5, 2, 100),
        'wind_direction_10m': np.random.normal(180, 30, 100)
    })
    
    # Generate EDA report
    report = eda.generate_eda_report(test_data, "test_data")
    print(f"EDA report status: {report['status']}")
    print(f"Visualizations generated: {list(report.get('artifacts', {}).keys())}")
    
    print("✅ EDA snapshot test completed!")
