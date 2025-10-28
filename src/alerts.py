"""
AQI Alerting System
Monitors AQI values and predictions against thresholds and generates alerts
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AQIAlertSystem:
    """
    AQI Alerting System that monitors current and predicted AQI values
    against predefined thresholds and generates alerts
    """
    
    def __init__(self):
        # AQI thresholds based on EPA standards
        self.aqi_thresholds = {
            AlertSeverity.LOW: 50,      # Good
            AlertSeverity.MODERATE: 100, # Moderate
            AlertSeverity.HIGH: 150,     # Unhealthy for Sensitive Groups
            AlertSeverity.CRITICAL: 200 # Unhealthy
        }
        
        # Alert storage
        self.alerts_dir = "alerts"
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Alert history file
        self.alerts_file = os.path.join(self.alerts_dir, "alert_history.json")
        
        # Load existing alerts
        self.alert_history = self._load_alert_history()
    
    def evaluate_alerts(self, current_aqi: float, predictions: Dict[int, Dict[str, float]], 
                       timestamp: datetime = None) -> List[Dict[str, Any]]:
        """
        Evaluate alerts based on current AQI and predictions
        
        Args:
            current_aqi: Current AQI value
            predictions: Dictionary of horizon -> model -> prediction
            timestamp: Timestamp for the evaluation
            
        Returns:
            List of generated alerts
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        alerts = []
        
        try:
            # Check current AQI
            current_alerts = self._check_aqi_thresholds(current_aqi, timestamp, "current")
            alerts.extend(current_alerts)
            
            # Check predictions for each horizon
            for horizon, model_predictions in predictions.items():
                for model_name, prediction in model_predictions.items():
                    pred_alerts = self._check_aqi_thresholds(
                        prediction, timestamp, f"{horizon}h_{model_name}"
                    )
                    alerts.extend(pred_alerts)
            
            # Save alerts
            if alerts:
                self._save_alerts(alerts)
                logger.info(f"Generated {len(alerts)} alerts")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert evaluation failed: {str(e)}")
            return []
    
    def _check_aqi_thresholds(self, aqi_value: float, timestamp: datetime, 
                            source: str) -> List[Dict[str, Any]]:
        """Check AQI value against thresholds and generate alerts"""
        alerts = []
        
        try:
            # Determine severity based on AQI value
            severity = self._get_aqi_severity(aqi_value)
            
            # Only generate alert if AQI is above "Good" threshold (50)
            if aqi_value > self.aqi_thresholds[AlertSeverity.LOW]:
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'aqi_value': round(aqi_value, 1),
                    'severity': severity.value,
                    'source': source,
                    'message': self._generate_alert_message(aqi_value, severity, source),
                    'threshold_exceeded': self._get_exceeded_threshold(aqi_value),
                    'recommendations': self._get_recommendations(severity)
                }
                
                # Check if this is a new alert (avoid duplicates)
                if self._is_new_alert(alert):
                    alerts.append(alert)
                    logger.info(f"New {severity.value} alert: AQI {aqi_value:.1f} from {source}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Threshold check failed: {str(e)}")
            return []
    
    def _get_aqi_severity(self, aqi_value: float) -> AlertSeverity:
        """Determine alert severity based on AQI value"""
        if aqi_value >= self.aqi_thresholds[AlertSeverity.CRITICAL]:
            return AlertSeverity.CRITICAL
        elif aqi_value >= self.aqi_thresholds[AlertSeverity.HIGH]:
            return AlertSeverity.HIGH
        elif aqi_value >= self.aqi_thresholds[AlertSeverity.MODERATE]:
            return AlertSeverity.MODERATE
        else:
            return AlertSeverity.LOW
    
    def _get_exceeded_threshold(self, aqi_value: float) -> str:
        """Get the highest threshold exceeded"""
        if aqi_value >= self.aqi_thresholds[AlertSeverity.CRITICAL]:
            return "Critical (200+)"
        elif aqi_value >= self.aqi_thresholds[AlertSeverity.HIGH]:
            return "High (150+)"
        elif aqi_value >= self.aqi_thresholds[AlertSeverity.MODERATE]:
            return "Moderate (100+)"
        else:
            return "Low (50+)"
    
    def _generate_alert_message(self, aqi_value: float, severity: AlertSeverity, 
                               source: str) -> str:
        """Generate human-readable alert message"""
        severity_text = {
            AlertSeverity.LOW: "Moderate air quality concern",
            AlertSeverity.MODERATE: "Unhealthy air quality for sensitive groups",
            AlertSeverity.HIGH: "Unhealthy air quality",
            AlertSeverity.CRITICAL: "Very unhealthy air quality"
        }
        
        return f"{severity_text[severity]} detected: AQI {aqi_value:.1f} ({source})"
    
    def _get_recommendations(self, severity: AlertSeverity) -> List[str]:
        """Get recommendations based on alert severity"""
        recommendations = {
            AlertSeverity.LOW: [
                "Consider reducing outdoor activities if you're sensitive to air pollution",
                "Monitor air quality updates"
            ],
            AlertSeverity.MODERATE: [
                "Sensitive groups should reduce outdoor activities",
                "Consider wearing a mask if spending time outdoors",
                "Keep windows closed if possible"
            ],
            AlertSeverity.HIGH: [
                "Everyone should reduce outdoor activities",
                "Avoid strenuous outdoor exercise",
                "Keep windows and doors closed",
                "Use air purifiers if available"
            ],
            AlertSeverity.CRITICAL: [
                "Avoid all outdoor activities",
                "Stay indoors with windows and doors closed",
                "Use air purifiers",
                "Consider relocating if possible"
            ]
        }
        
        return recommendations[severity]
    
    def _is_new_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if this is a new alert (avoid duplicates)"""
        try:
            # Check if similar alert exists in recent history (within 1 hour)
            recent_time = datetime.now() - timedelta(hours=1)
            
            for existing_alert in self.alert_history:
                existing_time = datetime.fromisoformat(existing_alert['timestamp'])
                
                if (existing_time > recent_time and 
                    existing_alert['source'] == alert['source'] and
                    existing_alert['severity'] == alert['severity'] and
                    abs(existing_alert['aqi_value'] - alert['aqi_value']) < 5):  # Similar AQI
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Alert duplicate check failed: {str(e)}")
            return True
    
    def _save_alerts(self, alerts: List[Dict[str, Any]]):
        """Save alerts to history"""
        try:
            # Add to history
            self.alert_history.extend(alerts)
            
            # Keep only last 1000 alerts to prevent file from growing too large
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Save to file
            with open(self.alerts_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            
            logger.info(f"Saved {len(alerts)} alerts to history")
            
        except Exception as e:
            logger.error(f"Failed to save alerts: {str(e)}")
    
    def _load_alert_history(self) -> List[Dict[str, Any]]:
        """Load alert history from file"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to load alert history: {str(e)}")
            return []
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_alerts = []
            for alert in self.alert_history:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time > cutoff_time:
                    recent_alerts.append(alert)
            
            return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {str(e)}")
            return []
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours"""
        try:
            recent_alerts = self.get_recent_alerts(hours)
            
            if not recent_alerts:
                return {
                    'total_alerts': 0,
                    'severity_counts': {},
                    'latest_alert': None,
                    'highest_aqi': None
                }
            
            # Count by severity
            severity_counts = {}
            for alert in recent_alerts:
                severity = alert['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Find highest AQI
            highest_aqi = max(recent_alerts, key=lambda x: x['aqi_value'])
            
            return {
                'total_alerts': len(recent_alerts),
                'severity_counts': severity_counts,
                'latest_alert': recent_alerts[0],
                'highest_aqi': highest_aqi,
                'time_range': f"Last {hours} hours"
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert summary: {str(e)}")
            return {}
    
    def send_notification(self, alert: Dict[str, Any], notification_type: str = "log") -> bool:
        """
        Send notification for critical alerts
        
        Args:
            alert: Alert dictionary
            notification_type: Type of notification (log, email, webhook)
            
        Returns:
            True if notification sent successfully
        """
        try:
            if notification_type == "log":
                logger.warning(f"ALERT: {alert['message']}")
                return True
            
            elif notification_type == "email":
                # Placeholder for email notification
                logger.info(f"Email notification would be sent: {alert['message']}")
                return True
            
            elif notification_type == "webhook":
                # Placeholder for webhook notification
                logger.info(f"Webhook notification would be sent: {alert['message']}")
                return True
            
            else:
                logger.warning(f"Unknown notification type: {notification_type}")
                return False
                
        except Exception as e:
            logger.error(f"Notification failed: {str(e)}")
            return False
    
    def run_alert_check(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete alert check on current data and model predictions
        
        Args:
            data: Current data DataFrame
            models: Dictionary of trained models
            
        Returns:
            Alert check results
        """
        try:
            logger.info("Running alert check...")
            
            # Get current AQI (latest value)
            if 'aqi' in data.columns and not data.empty:
                current_aqi = data['aqi'].iloc[-1]
            else:
                logger.warning("No current AQI data available")
                return {'alerts': [], 'summary': {}}
            
            # Generate predictions for alert evaluation
            predictions = {}
            
            # Simple prediction for alerting (using latest features)
            if not data.empty and models:
                latest_features = data.iloc[-1:].select_dtypes(include=[np.number])
                
                # Remove target columns
                feature_cols = [c for c in latest_features.columns 
                              if c not in ['aqi', 'aqi_6h_ahead']]
                
                if feature_cols:
                    X_latest = latest_features[feature_cols]
                    
                    # Generate predictions for different horizons
                    for horizon in [1, 6, 12, 24]:
                        predictions[horizon] = {}
                        for model_name, model in models.items():
                            try:
                                pred = model.predict(X_latest)[0]
                                predictions[horizon][model_name] = pred
                            except Exception as e:
                                logger.warning(f"Prediction failed for {model_name}: {str(e)}")
                                predictions[horizon][model_name] = current_aqi  # Fallback
            
            # Evaluate alerts
            alerts = self.evaluate_alerts(current_aqi, predictions)
            
            # Send notifications for critical alerts
            for alert in alerts:
                if alert['severity'] == 'critical':
                    self.send_notification(alert, "log")
            
            # Get summary
            summary = self.get_alert_summary(24)
            
            logger.info(f"Alert check completed: {len(alerts)} new alerts")
            
            return {
                'alerts': alerts,
                'summary': summary,
                'current_aqi': current_aqi,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Alert check failed: {str(e)}")
            return {'alerts': [], 'summary': {}, 'error': str(e)}


def run_alert_system(data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete alerting system
    
    Args:
        data: Current data DataFrame
        models: Dictionary of trained models
        
    Returns:
        Alert system results
    """
    try:
        alert_system = AQIAlertSystem()
        results = alert_system.run_alert_check(data, models)
        return results
        
    except Exception as e:
        logger.error(f"Alert system failed: {str(e)}")
        return {'alerts': [], 'summary': {}, 'error': str(e)}


if __name__ == "__main__":
    # Test alert system
    alert_system = AQIAlertSystem()
    
    # Test with sample data
    test_alerts = alert_system.evaluate_alerts(
        current_aqi=120.5,
        predictions={
            1: {'random_forest': 125.0, 'ridge_regression': 118.0},
            6: {'random_forest': 130.0, 'ridge_regression': 122.0}
        }
    )
    
    print(f"Generated {len(test_alerts)} test alerts")
    for alert in test_alerts:
        print(f"- {alert['message']}")
    
    # Test summary
    summary = alert_system.get_alert_summary(24)
    print(f"Alert summary: {summary}")
