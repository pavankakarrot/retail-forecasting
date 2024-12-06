# src/utils.py

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> logging.Logger:
   """Configure logging for the project."""
   # Create logger
   logger = logging.getLogger("retail_forecast")
   
   # Prevent duplicate handlers 
   if logger.handlers:
       return logger
       
   # Set level
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
       """Initialize DataLoader with data directory path."""
       self.data_dir = Path(data_dir)
       if not self.data_dir.exists():
           raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
           
   def load_data(self, filename: str) -> pd.DataFrame:
       """Load data from specified file with flexible path handling."""
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
           
           raise FileNotFoundError(f"Could not find {filename} in any of the expected locations: {[str(p) for p in potential_paths]}")
           
       except Exception as e:
           logger.error(f"Error loading data: {str(e)}")
           raise

   def validate_data(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
       """Validate DataFrame has required columns and correct data types."""
       try:
           # Check for required columns
           missing_cols = set(required_cols) - set(df.columns)
           if missing_cols:
               logger.error(f"Missing required columns: {missing_cols}")
               return False
           
           # Check for empty DataFrame
           if df.empty:
               logger.error("DataFrame is empty")
               return False
               
           # Check for null values in required columns
           null_counts = df[required_cols].isnull().sum()
           if null_counts.any():
               logger.warning(f"Null values found in columns: \n{null_counts[null_counts > 0]}")
               
           return True
           
       except Exception as e:
           logger.error(f"Error during DataFrame validation: {str(e)}")
           return False

   def save_data(self, df: pd.DataFrame, filename: str, subdir: str = "processed") -> None:
       """Save DataFrame to specified location."""
       try:
           # Create subdirectory if it doesn't exist
           save_dir = self.data_dir / subdir
           save_dir.mkdir(parents=True, exist_ok=True)
           
           # Save file
           save_path = save_dir / filename
           df.to_csv(save_path, index=False)
           logger.info(f"Successfully saved data to {save_path}")
           
       except Exception as e:
           logger.error(f"Error saving data: {str(e)}")
           raise
