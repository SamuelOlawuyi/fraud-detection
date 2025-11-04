"""Download datasets from Kaggle."""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import RAW_DATA_DIR, settings
from src.utils.logger import get_logger

logger = get_logger()

def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    if settings.kaggle_username and settings.kaggle_key:
        os.environ['KAGGLE_USERNAME'] = settings.kaggle_username
        os.environ['KAGGLE_KEY'] = settings.kaggle_key
        logger.info("Kaggle credentials configured from environment")
    else:
        logger.warning("Kaggle credentials not found in environment. Using ~/.kaggle/kaggle.json")


def download_creditcard_dataset():
    """Download Credit Card Fraud Detection dataset from Kaggle."""
    try:
        setup_kaggle_credentials()
        
        # Import kaggle AFTER setting credentials
        import kaggle
        
        # Ensure directory exists
        try:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            pass  # Directory already exists
        
        logger.info(f"Downloading dataset: {settings.creditcard_dataset}")
        logger.info(f"Saving to: {RAW_DATA_DIR}")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            settings.creditcard_dataset,
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        
        logger.info(f"Dataset downloaded successfully to {RAW_DATA_DIR}")
        
        # List downloaded files
        files = list(RAW_DATA_DIR.glob("*.csv"))
        logger.info(f"Downloaded files: {[f.name for f in files]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.info("Please ensure Kaggle API credentials are properly configured")
        logger.info("Visit: https://www.kaggle.com/docs/api")
        return False


def verify_dataset():
    """Verify that the dataset exists."""
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        logger.warning("No CSV files found in raw data directory")
        return False
    
    logger.info(f"Found {len(csv_files)} CSV file(s) in {RAW_DATA_DIR}")
    for file in csv_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {file.name}: {size_mb:.2f} MB")
    
    return True


if __name__ == "__main__":
    logger.info("Starting dataset download...")
    
    if not verify_dataset():
        logger.info("Dataset not found locally. Downloading from Kaggle...")
        success = download_creditcard_dataset()
        
        if success:
            logger.info("✓ Dataset download completed successfully")
        else:
            logger.error("✗ Dataset download failed")
    else:
        logger.info("✓ Dataset already exists locally")