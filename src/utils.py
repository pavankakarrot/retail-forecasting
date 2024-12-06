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
       try:
           file_path = self.data_dir / "raw" / filename
        
           # Check file exists and size
           if file_path.exists():
               size = file_path.stat().st_size
               logger.info(f"File size: {size} bytes")
           else:
               raise FileNotFoundError(f"File not found: {file_path}")

           # Read file content directly first
           with open(file_path, 'rb') as f:
               content = f.read(1000)  # Read first 1000 bytes
               logger.info(f"File preview: {content[:200]}")

           # Try different pandas read options
           df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
           logger.info(f"Loaded DataFrame shape: {df.shape}")
           logger.info(f"Columns: {df.columns.tolist()}")
        
           return df
        
    except Exception as e:
        logger.error(f"Error details: {str(e)}")
        raise

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
