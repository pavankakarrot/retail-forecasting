# src/preprocessing.py
# Purpose: Handles all data preprocessing and feature engineering tasks

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import logging

# Get the logger
logger = logging.getLogger("retail_forecast")

class RetailPreprocessor:
    """A class to handle all preprocessing steps for retail forecasting data."""
    
    def __init__(self):
        """Initialize the preprocessor with necessary encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = ['Country', 'Category', 'PaymentMethod']
        self.numeric_columns = ['Quantity', 'UnitPrice', 'TotalAmount']
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline that coordinates all transformation steps.
        
        Args:
            df: Raw input DataFrame
        
        Returns:
            Preprocessed DataFrame ready for modeling
        """
        try:
            logger.info("Starting data preprocessing pipeline...")
            
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Execute preprocessing steps in sequence
            processed_df = self.clean_data(processed_df)
            processed_df = self.create_time_features(processed_df)
            processed_df = self.create_lag_features(processed_df)
            processed_df = self.encode_categorical_features(processed_df)
            
            logger.info("Data preprocessing completed successfully")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data by handling missing values and outliers."""
        logger.info("Starting data cleaning...")
        df_cleaned = df.copy()
        
        # Handle missing values
        for col in self.numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        for col in self.categorical_columns:
            if col in df_cleaned.columns:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        logger.info("Data cleaning completed")
        return df_cleaned

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from datetime columns."""
        logger.info("Creating time-based features...")
        df_time = df.copy()
        
        if 'InvoiceDate' in df_time.columns:
            df_time['year'] = df_time['InvoiceDate'].dt.year
            df_time['month'] = df_time['InvoiceDate'].dt.month
            df_time['day'] = df_time['InvoiceDate'].dt.day
            df_time['day_of_week'] = df_time['InvoiceDate'].dt.dayofweek
            df_time['hour'] = df_time['InvoiceDate'].dt.hour
        
        logger.info("Time features created successfully")
        return df_time

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col])
        
        logger.info("Categorical features encoded successfully")
        return df_encoded
