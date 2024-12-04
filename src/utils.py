# src/utils.py

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
import yaml
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
    file_handler = logging.FileHandler(f"logs/retail_forecast_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class DataLoader:
    """
    Class to handle all data loading operations.
    """
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
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Determine file type and load accordingly
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded CSV file: {filename}")
                return df
            except Exception as e:
                logger.error(f"Error loading CSV file {filename}: {str(e)}")
                raise
                
        elif file_path.suffix in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file_path)
                logger.info(f"Successfully loaded Excel file: {filename}")
                return df
            except Exception as e:
                logger.error(f"Error loading Excel file {filename}: {str(e)}")
                raise
                
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame has required columns and correct data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
        
    # Basic data validation
    try:
        # Check for empty DataFrame
        if df.empty:
            logger.error("DataFrame is empty")
            return False
            
        # Check for null values in required columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found in columns: \n{null_counts[null_counts > 0]}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during DataFrame validation: {str(e)}")
        return False

def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate key business metrics from the data.
    
    Args:
        df: DataFrame containing sales data
        
    Returns:
        Dictionary containing calculated metrics
    """
    try:
        metrics = {
            'total_sales': df['TotalAmount'].sum(),
            'average_transaction_value': df['TotalAmount'].mean(),
            'transaction_count': len(df),
            'unique_customers': df['CustomerID'].nunique(),
            'average_items_per_transaction': df['Quantity'].mean()
        }
        
        logger.info("Successfully calculated business metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise
