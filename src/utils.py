# src/utils.py
# Purpose: Contains utility functions and classes for data loading and logging

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the project.
    Creates both console and file handlers for comprehensive logging.
    
    Args:
        log_level: Desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger("retail_forecast")
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
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
    """Class to handle all data loading operations."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = Path(data_dir)
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from specified file with flexible path handling.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            pandas DataFrame containing the loaded data
        """
        try:
            # Handle path for both local and Streamlit Cloud environments
            potential_paths = [
                Path(filename),  # Direct path
                self.data_dir / filename,  # data/filename
                self.data_dir / "raw" / filename,  # data/raw/filename
                self.data_dir / "processed" / filename  # data/processed/filename
            ]
            
            # Try each path until we find the file
            for file_path in potential_paths:
                if file_path.exists():
                    logger.info(f"Loading data from: {file_path}")
                    df = pd.read_csv(file_path)
                    
                    # Convert date column if it exists
                    if 'InvoiceDate' in df.columns:
                        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                    
                    return df
            
            raise FileNotFoundError(f"Could not find {filename} in any of the expected locations")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
