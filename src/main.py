"""
Simplified Main Entry Point for Air Quality Index Forecasting
Unified pipeline with minimal commands
"""

import argparse
import logging
import config
from src.pipeline import run_unified_pipeline

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_dashboard():
    """Run Streamlit dashboard"""
    try:
        logger.info("Starting Streamlit dashboard...")
        
        import subprocess
        import sys
        
        # Run dashboard
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Dashboard started successfully")
            return True
        else:
            logger.error(f"Dashboard failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Dashboard failed: {str(e)}")
        return False


def test_connections():
    """Test basic connections"""
    try:
        logger.info("Testing connections...")
        
        # Test Hopsworks connection
        from src.hopsworks_client import HopsworksClient
        client = HopsworksClient()
        
        if client.connect():
            logger.info("Hopsworks connection: OK")
        else:
            logger.error("Hopsworks connection: FAILED")
            return False
        
        # Test data fetcher
        from src.data_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        
        # Test with small date range
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        data = fetcher.fetch_combined_historical_data(
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        if data is not None and not data.empty:
            logger.info("Data fetcher: OK")
        else:
            logger.error("Data fetcher: FAILED")
            return False
        
        logger.info("All connections tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False


def main():
    """Main entry point with simplified commands"""
    parser = argparse.ArgumentParser(description="Air Quality Index Forecasting - Unified Pipeline")
    parser.add_argument("command", choices=["setup", "update", "dashboard", "test"], 
                       help="Command to run: setup (initial), update (daily), dashboard, test")
    
    args = parser.parse_args()
    
    logger.info("=== Air Quality Index Forecasting Pipeline ===")
    logger.info(f"Command: {args.command}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.command == "dashboard":
            success = run_dashboard()
        elif args.command == "test":
            success = test_connections()
        else:
            # Run unified pipeline
            success = run_unified_pipeline(mode=args.command)
        
        if success:
            logger.info(f"{args.command} completed successfully!")
        else:
            logger.error(f"{args.command} failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {str(e)}")
        return False


if __name__ == "__main__":
    from datetime import datetime
    main()