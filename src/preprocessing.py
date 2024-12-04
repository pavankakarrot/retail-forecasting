import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from datetime import datetime
from .utils import logger

class RetailPreprocessor:
    """
    A class to handle all preprocessing steps for retail forecasting data.
    This includes feature engineering, data cleaning, and transformation.
    """
    def __init__(self):
        """Initialize the preprocessor with necessary encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = ['Country', 'Category', 'PaymentMethod', 'ShipmentProvider']
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
            processed_df = self.create_rolling_features(processed_df)
            processed_df = self.encode_categorical_features(processed_df)
            
            logger.info("Data preprocessing completed successfully")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, outliers, and data type conversions.
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_cleaned = df.copy()
        
        # Convert date columns
        try:
            df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
        except Exception as e:
            logger.error(f"Error converting date column: {str(e)}")
            raise
            
        # Handle missing values
        for col in self.numeric_columns:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            
        for col in self.categorical_columns:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            
        # Remove outliers using IQR method
        for col in self.numeric_columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
            
        logger.info("Data cleaning completed")
        return df_cleaned

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime columns.
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with additional time-based features
        """
        logger.info("Creating time-based features...")
        df_time = df.copy()
        
        # Extract basic time components
        df_time['year'] = df_time['InvoiceDate'].dt.year
        df_time['month'] = df_time['InvoiceDate'].dt.month
        df_time['day'] = df_time['InvoiceDate'].dt.day
        df_time['day_of_week'] = df_time['InvoiceDate'].dt.dayofweek
        df_time['hour'] = df_time['InvoiceDate'].dt.hour
        
        # Create cyclical features for temporal variables
        df_time['month_sin'] = np.sin(2 * np.pi * df_time['month']/12)
        df_time['month_cos'] = np.cos(2 * np.pi * df_time['month']/12)
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour']/24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour']/24)
        df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week']/7)
        df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week']/7)
        
        # Add is_weekend flag
        df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info("Time features created successfully")
        return df_time

    def create_lag_features(self, df: pd.DataFrame, 
                          target_col: str = 'TotalAmount',
                          lag_periods: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lagged features for the target variable.
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lag_periods: List of periods to lag
            
        Returns:
            DataFrame with additional lag features
        """
        logger.info(f"Creating lag features for {target_col}...")
        df_lag = df.copy()
        
        for lag in lag_periods:
            df_lag[f'{target_col}_lag_{lag}'] = df_lag.groupby(['year', 'month'])[target_col].shift(lag)
            
        logger.info("Lag features created successfully")
        return df_lag

    def create_rolling_features(self, df: pd.DataFrame,
                              target_col: str = 'TotalAmount',
                              windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window features for the target variable.
        
        Args:
            df: Input DataFrame
            target_col: Column to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with additional rolling features
        """
        logger.info(f"Creating rolling features for {target_col}...")
        df_roll = df.copy()
        
        for window in windows:
            df_roll[f'{target_col}_rolling_mean_{window}'] = (
                df_roll.groupby(['year', 'month'])[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df_roll[f'{target_col}_rolling_std_{window}'] = (
                df_roll.groupby(['year', 'month'])[target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
        logger.info("Rolling features created successfully")
        return df_roll

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col])
                
        logger.info("Categorical features encoded successfully")
        return df_encoded
