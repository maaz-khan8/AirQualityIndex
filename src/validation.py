"""
Data Quality Validation Module
Validates data quality, schema, ranges, and freshness before processing
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data quality validator that checks schema, ranges, nulls, and freshness
    """
    
    def __init__(self):
        # Expected schema for air quality data
        self.expected_schema = {
            'timestamp': 'datetime64[ns]',
            'city': 'object',
            'latitude': 'float64',
            'longitude': 'float64',
            'aqi': 'float64',
            'pm2_5': 'float64',
            'pm10': 'float64',
            'o3': 'float64',
            'no2': 'float64',
            'so2': 'float64',
            'co': 'float64',
            'temperature_2m': 'float64',
            'relative_humidity_2m': 'float64',
            'pressure_msl': 'float64',
            'wind_speed_10m': 'float64',
            'wind_direction_10m': 'float64'
        }
        
        # Valid ranges for different variables
        self.valid_ranges = {
            'aqi': (0, 500),  # AQI range
            'pm2_5': (0, 500),  # μg/m³
            'pm10': (0, 1000),  # μg/m³
            'o3': (0, 500),  # μg/m³
            'no2': (0, 500),  # μg/m³
            'so2': (0, 1000),  # μg/m³
            'co': (0, 50),  # mg/m³
            'temperature_2m': (-50, 60),  # °C
            'relative_humidity_2m': (0, 100),  # %
            'pressure_msl': (800, 1200),  # hPa
            'wind_speed_10m': (0, 100),  # m/s
            'wind_direction_10m': (0, 360)  # degrees
        }
        
        # Critical columns that must not be null
        self.critical_columns = ['timestamp', 'aqi', 'pm2_5', 'pm10']
        
        # Validation results storage
        self.validation_dir = "validation"
        os.makedirs(self.validation_dir, exist_ok=True)
        
    def validate_data(self, df: pd.DataFrame, data_source: str = "unknown") -> Dict[str, Any]:
        """
        Perform comprehensive data validation
        
        Args:
            df: DataFrame to validate
            data_source: Source identifier for the data
            
        Returns:
            Validation results dictionary
        """
        try:
            logger.info(f"Starting data validation for {data_source}")
            
            validation_results = {
                'data_source': data_source,
                'validation_timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'total_columns': len(df.columns),
                'schema_check': {},
                'range_check': {},
                'null_check': {},
                'freshness_check': {},
                'overall_status': 'PASS',
                'issues': [],
                'recommendations': []
            }
            
            if df.empty:
                validation_results['overall_status'] = 'FAIL'
                validation_results['issues'].append("Empty dataset")
                return validation_results
            
            # 1. Schema validation
            schema_results = self._validate_schema(df)
            validation_results['schema_check'] = schema_results
            
            # 2. Range validation
            range_results = self._validate_ranges(df)
            validation_results['range_check'] = range_results
            
            # 3. Null validation
            null_results = self._validate_nulls(df)
            validation_results['null_check'] = null_results
            
            # 4. Freshness validation
            freshness_results = self._validate_freshness(df)
            validation_results['freshness_check'] = freshness_results
            
            # Determine overall status
            validation_results = self._determine_overall_status(validation_results)
            
            # Save validation results
            self._save_validation_results(validation_results)
            
            logger.info(f"Data validation completed: {validation_results['overall_status']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                'data_source': data_source,
                'validation_timestamp': datetime.now().isoformat(),
                'overall_status': 'ERROR',
                'error': str(e)
            }
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema"""
        try:
            schema_results = {
                'status': 'PASS',
                'missing_columns': [],
                'extra_columns': [],
                'type_mismatches': [],
                'column_count': len(df.columns)
            }
            
            # Check for missing columns
            expected_cols = set(self.expected_schema.keys())
            actual_cols = set(df.columns)
            
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            
            if missing_cols:
                schema_results['missing_columns'] = list(missing_cols)
                schema_results['status'] = 'WARN'
                logger.warning(f"Missing columns: {missing_cols}")
            
            if extra_cols:
                schema_results['extra_columns'] = list(extra_cols)
                logger.info(f"Extra columns found: {extra_cols}")
            
            # Check data types for existing columns
            for col, expected_type in self.expected_schema.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type:
                        schema_results['type_mismatches'].append({
                            'column': col,
                            'expected': expected_type,
                            'actual': actual_type
                        })
                        schema_results['status'] = 'WARN'
            
            return schema_results
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data ranges"""
        try:
            range_results = {
                'status': 'PASS',
                'out_of_range': {},
                'total_violations': 0
            }
            
            for col, (min_val, max_val) in self.valid_ranges.items():
                if col in df.columns:
                    # Check for values outside valid range
                    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                    
                    if not out_of_range.empty:
                        range_results['out_of_range'][col] = {
                            'min_valid': min_val,
                            'max_valid': max_val,
                            'violations': int(len(out_of_range)),
                            'min_actual': float(df[col].min()),
                            'max_actual': float(df[col].max()),
                            'violation_percentage': float((len(out_of_range) / len(df)) * 100)
                        }
                        range_results['total_violations'] += int(len(out_of_range))
                        
                        violation_percentage = float((len(out_of_range) / len(df)) * 100)
                        if violation_percentage > 5:  # More than 5% violations
                            range_results['status'] = 'FAIL'
                        elif violation_percentage > 1:  # More than 1% violations
                            range_results['status'] = 'WARN'
            
            return range_results
            
        except Exception as e:
            logger.error(f"Range validation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_nulls(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate null values"""
        try:
            null_results = {
                'status': 'PASS',
                'null_counts': {},
                'critical_nulls': {},
                'total_nulls': int(df.isnull().sum().sum())
            }
            
            # Check null counts for all columns
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                
                null_results['null_counts'][col] = {
                    'count': int(null_count),
                    'percentage': round(null_percentage, 2)
                }
                
                # Check critical columns
                if col in self.critical_columns and null_count > 0:
                    null_results['critical_nulls'][col] = {
                        'count': int(null_count),
                        'percentage': round(null_percentage, 2)
                    }
                    null_results['status'] = 'FAIL'
                    logger.error(f"Critical column {col} has {null_count} null values")
                
                # Check for high null percentage
                elif null_percentage > 20:  # More than 20% nulls
                    null_results['status'] = 'WARN'
                    logger.warning(f"Column {col} has {null_percentage:.1f}% null values")
            
            return null_results
            
        except Exception as e:
            logger.error(f"Null validation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_freshness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data freshness"""
        try:
            freshness_results = {
                'status': 'PASS',
                'latest_timestamp': None,
                'oldest_timestamp': None,
                'data_age_hours': None,
                'time_gaps': []
            }
            
            if 'timestamp' not in df.columns:
                freshness_results['status'] = 'WARN'
                freshness_results['issues'] = ['No timestamp column found']
                return freshness_results
            
            # Convert timestamp to datetime if needed
            timestamps = pd.to_datetime(df['timestamp'])
            
            latest_time = timestamps.max()
            oldest_time = timestamps.min()
            current_time = datetime.now()
            
            freshness_results['latest_timestamp'] = latest_time.isoformat()
            freshness_results['oldest_timestamp'] = oldest_time.isoformat()
            
            # Calculate data age
            data_age = current_time - latest_time
            freshness_results['data_age_hours'] = data_age.total_seconds() / 3600
            
            # Check data freshness
            if data_age.total_seconds() > 24 * 3600:  # Older than 24 hours
                freshness_results['status'] = 'WARN'
                logger.warning(f"Data is {data_age.total_seconds() / 3600:.1f} hours old")
            
            # Check for time gaps
            timestamps_sorted = timestamps.sort_values()
            time_diffs = timestamps_sorted.diff()
            
            # Look for gaps larger than 2 hours
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
            if not large_gaps.empty:
                freshness_results['time_gaps'] = [
                    {
                        'gap_start': timestamps_sorted.iloc[i-1].isoformat(),
                        'gap_end': timestamps_sorted.iloc[i].isoformat(),
                        'gap_hours': gap.total_seconds() / 3600
                    }
                    for i, gap in large_gaps.items()
                ]
                freshness_results['status'] = 'WARN'
                logger.warning(f"Found {len(large_gaps)} time gaps larger than 2 hours")
            
            return freshness_results
            
        except Exception as e:
            logger.error(f"Freshness validation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall validation status"""
        try:
            # Check individual validation statuses
            statuses = [
                results['schema_check'].get('status', 'PASS'),
                results['range_check'].get('status', 'PASS'),
                results['null_check'].get('status', 'PASS'),
                results['freshness_check'].get('status', 'PASS')
            ]
            
            # Determine overall status
            if 'FAIL' in statuses:
                results['overall_status'] = 'FAIL'
                results['issues'].append("Critical validation failures detected")
            elif 'WARN' in statuses:
                results['overall_status'] = 'WARN'
                results['issues'].append("Validation warnings detected")
            else:
                results['overall_status'] = 'PASS'
            
            # Add recommendations
            if results['overall_status'] != 'PASS':
                results['recommendations'].extend(self._generate_recommendations(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Status determination failed: {str(e)}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
            return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Schema recommendations
        if results['schema_check'].get('missing_columns'):
            recommendations.append("Add missing columns to data source")
        
        # Range recommendations
        range_check = results['range_check']
        if range_check.get('out_of_range'):
            recommendations.append("Review data collection process for out-of-range values")
        
        # Null recommendations
        null_check = results['null_check']
        if null_check.get('critical_nulls'):
            recommendations.append("Fix null values in critical columns before processing")
        
        # Freshness recommendations
        freshness_check = results['freshness_check']
        if freshness_check.get('data_age_hours', 0) > 24:
            recommendations.append("Update data collection frequency for fresher data")
        
        return recommendations
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"validation_results_{timestamp}.json"
            filepath = os.path.join(self.validation_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Validation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {str(e)}")
    
    def get_latest_validation_summary(self) -> Optional[Dict[str, Any]]:
        """Get the latest validation summary"""
        try:
            if not os.path.exists(self.validation_dir):
                return None
            
            # Find the latest validation file
            validation_files = [f for f in os.listdir(self.validation_dir) 
                              if f.startswith('validation_results_') and f.endswith('.json')]
            
            if not validation_files:
                return None
            
            latest_file = sorted(validation_files)[-1]
            filepath = os.path.join(self.validation_dir, latest_file)
            
            with open(filepath, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load latest validation summary: {str(e)}")
            return None


def validate_data_quality(df: pd.DataFrame, data_source: str = "unknown") -> Dict[str, Any]:
    """
    Convenience function to validate data quality
    
    Args:
        df: DataFrame to validate
        data_source: Source identifier
        
    Returns:
        Validation results dictionary
    """
    try:
        validator = DataValidator()
        return validator.validate_data(df, data_source)
    except Exception as e:
        logger.error(f"Data quality validation failed: {str(e)}")
        return {'overall_status': 'ERROR', 'error': str(e)}


if __name__ == "__main__":
    # Test data validator
    validator = DataValidator()
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-10-28', periods=100, freq='H'),
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
    
    # Validate test data
    results = validator.validate_data(test_data, "test_data")
    print(f"Validation status: {results['overall_status']}")
    print(f"Total records: {results['total_records']}")
    
    print("✅ Data validation test completed!")
