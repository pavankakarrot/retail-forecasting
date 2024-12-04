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
    """Class to handle all data loading operations."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from specified file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Pandas DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        try:
            # For Streamlit Cloud, use direct file path
            file_path = Path(filename) if str(filename).startswith('/') else self.data_dir / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and load accordingly
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded CSV file: {filename}")
                return df
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                logger.info(f"Successfully loaded Excel file: {filename}")
                return df
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            raise
