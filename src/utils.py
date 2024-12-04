# src/utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
from datetime import datetime

# Set up logging configuration
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_level: Desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger object configured with the specified settings
    """
    # Create a logger
    logger = logging.getLogger("retail_forecast")
    
    # Set logging level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create handlers
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class DataLoader:
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed features and target data.
        
        Returns:
            Tuple containing:
            - Features DataFrame
            - Target DataFrame
        """
        try:
            # Load features
            features_path = self.data_dir / "processed" / "processed_features.csv"
            if not features_path.exists():
                raise FileNotFoundError(f"Processed features not found: {features_path}")
            
            features_df = pd.read_csv(features_path)
            
            # Load target
            target_path = self.data_dir / "processed" / "target_data.csv"
            if not target_path.exists():
                raise FileNotFoundError(f"Target data not found: {target_path}")
                
            target_df = pd.read_csv(target_path)
            
            logger.info("Successfully loaded processed data")
            return features_df, target_df
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
