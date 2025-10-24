"""
Hopsworks Job Configuration for Automated Retraining
Configuration for scheduling daily model retraining on Hopsworks platform
"""

import os
from datetime import datetime, timedelta

# Hopsworks Job Configuration
JOB_CONFIG = {
    "name": "aqi_daily_retraining",
    "description": "Daily automated retraining of AQI forecasting models",
    "schedule": {
        "type": "cron",
        "expression": "0 2 * * *",  # Daily at 2 AM UTC
        "timezone": "UTC"
    },
    "resources": {
        "cpu": "2",
        "memory": "4Gi",
        "gpu": "0"
    },
    "environment": {
        "python_version": "3.12",
        "packages": [
            "hopsworks>=3.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
            "requests>=2.28.0",
            "plotly>=5.0.0"
        ]
    },
    "execution": {
        "command": "python src/retrain.py",
        "working_directory": "/",
        "timeout": "7200"  # 2 hours timeout
    },
    "notifications": {
        "on_success": True,
        "on_failure": True,
        "email": os.getenv("NOTIFICATION_EMAIL", "your-email@example.com")
    },
    "retry": {
        "max_attempts": 3,
        "delay": 300  # 5 minutes between retries
    }
}

# Environment Variables for the Job
ENVIRONMENT_VARIABLES = {
    "HOPSWORKS_API_KEY": os.getenv("HOPSWORKS_API_KEY", ""),
    "HOPSWORKS_PROJECT_NAME": os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_maazkhan"),
    "LOG_LEVEL": "INFO",
    "NOTIFICATION_EMAIL": os.getenv("NOTIFICATION_EMAIL", ""),
    "PYTHONPATH": "/"
}

# Job Dependencies
JOB_DEPENDENCIES = [
    "src/retrain.py",
    "src/data_fetcher.py",
    "src/feature_engineering.py", 
    "src/training.py",
    "src/hopsworks_client.py",
    "src/aqi_calculator.py",
    "config.py",
    "requirements.txt"
]

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics": [
        "execution_time",
        "model_performance",
        "data_quality",
        "deployment_success"
    ],
    "alerts": {
        "execution_failure": True,
        "performance_degradation": True,
        "data_quality_issues": True
    },
    "retention": {
        "logs_days": 30,
        "metrics_days": 90
    }
}

def get_job_config():
    """Get the complete job configuration"""
    return {
        "job": JOB_CONFIG,
        "environment": ENVIRONMENT_VARIABLES,
        "dependencies": JOB_DEPENDENCIES,
        "monitoring": MONITORING_CONFIG
    }

def print_setup_instructions():
    """Print setup instructions for Hopsworks Job"""
    print("""
🚀 HOPSWORKS JOB SETUP INSTRUCTIONS
====================================

1. CREATE JOB IN HOPSWORKS UI:
   - Go to Hopsworks UI → Jobs
   - Click "Create Job"
   - Name: aqi_daily_retraining
   - Type: Python

2. CONFIGURE JOB SETTINGS:
   - Schedule: Cron expression: 0 2 * * * (Daily at 2 AM UTC)
   - Resources: 2 CPU, 4GB RAM
   - Timeout: 2 hours
   - Python Version: 3.12

3. SET ENVIRONMENT VARIABLES:
   - HOPSWORKS_API_KEY: Your Hopsworks API key
   - HOPSWORKS_PROJECT_NAME: aqi_maazkhan
   - NOTIFICATION_EMAIL: Your email for alerts

4. UPLOAD CODE FILES:
   - Upload all files from src/ directory
   - Upload config.py and requirements.txt
   - Set working directory to project root

5. CONFIGURE NOTIFICATIONS:
   - Enable success notifications
   - Enable failure notifications
   - Set email for alerts

6. TEST THE JOB:
   - Run job manually first
   - Check logs for any issues
   - Verify model deployment

7. MONITOR EXECUTION:
   - Check job status daily
   - Review performance metrics
   - Monitor model performance

📧 NOTIFICATION SETUP:
   - Configure email alerts in Hopsworks
   - Set up monitoring dashboards
   - Track execution history

🔧 TROUBLESHOOTING:
   - Check job logs for errors
   - Verify API connections
   - Monitor resource usage
   - Review model performance

Next Steps:
1. Set up Hopsworks Job using these instructions
2. Test the job manually
3. Configure GitHub Actions monitoring
4. Set up alert notifications
""")

if __name__ == "__main__":
    print_setup_instructions()
