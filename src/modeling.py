# src/modeling.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import logging

class RetailForecaster:
    """
    A hybrid forecasting model combining Prophet for trend/seasonality
    and XGBoost for incorporating additional features.
    """
    def __init__(
        self,
        prophet_params: Optional[Dict] = None,
        xgboost_params: Optional[Dict] = None
    ):
        """
        Initialize the hybrid forecasting model.

        Args:
            prophet_params: Configuration parameters for Prophet model
            xgboost_params: Configuration parameters for XGBoost model
        """
        # Default Prophet parameters
        self.prophet_params = prophet_params or {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'seasonality_mode': 'multiplicative'
        }
        
        # Default XGBoost parameters
        self.xgboost_params = xgboost_params or {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': ['rmse', 'mae']
        }
        
        # Initialize models
        self.prophet_model = None
        self.xgboost_model = None
        
        # Store feature importance
        self.feature_importance = None

    def prepare_prophet_data(
        self, 
        df: pd.DataFrame,
        date_column: str,
        target_column: str
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet model.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            target_column: Name of target column

        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = df[[date_column, target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df

    def train_prophet(
        self,
        train_data: pd.DataFrame,
        date_column: str,
        target_column: str
    ) -> None:
        """
        Train the Prophet model component.

        Args:
            train_data: Training data
            date_column: Name of date column
            target_column: Name of target column
        """
        try:
            logger.info("Training Prophet model...")
            
            # Prepare data for Prophet
            prophet_data = self.prepare_prophet_data(
                train_data, 
                date_column, 
                target_column
            )
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(**self.prophet_params)
            self.prophet_model.fit(prophet_data)
            
            logger.info("Prophet model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            raise

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> None:
        """
        Train the XGBoost model component.

        Args:
            X_train: Training features
            y_train: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
        try:
            logger.info("Training XGBoost model...")
            
            # Initialize XGBoost model
            self.xgboost_model = xgb.XGBRegressor(**self.xgboost_params)
            
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if validation_data is not None:
                eval_set.append(validation_data)
            
            # Train model
            self.xgboost_model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Store feature importance
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.xgboost_model.feature_importances_
            })
            self.feature_importance = importance.sort_values(
                'importance', 
                ascending=False
            )
            
            logger.info("XGBoost model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            raise

    def make_predictions(
        self,
        future_dates: pd.DataFrame,
        feature_data: pd.DataFrame,
        prophet_weight: float = 0.4
    ) -> pd.DataFrame:
        """
        Generate hybrid predictions combining Prophet and XGBoost.

        Args:
            future_dates: DataFrame with future dates for Prophet
            feature_data: Feature DataFrame for XGBoost
            prophet_weight: Weight for Prophet predictions (0-1)

        Returns:
            DataFrame with hybrid predictions
        """
        try:
            logger.info("Generating hybrid predictions...")
            
            # Generate Prophet predictions
            prophet_predictions = self.prophet_model.predict(future_dates)
            
            # Generate XGBoost predictions
            xgb_predictions = self.xgboost_model.predict(feature_data)
            
            # Combine predictions
            final_predictions = (
                prophet_weight * prophet_predictions['yhat'] +
                (1 - prophet_weight) * xgb_predictions
            )
            
            # Create results DataFrame
            results = pd.DataFrame({
                'date': prophet_predictions['ds'],
                'prophet_pred': prophet_predictions['yhat'],
                'xgb_pred': xgb_predictions,
                'hybrid_pred': final_predictions
            })
            
            logger.info("Hybrid predictions generated successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate various evaluation metrics for predictions.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance from XGBoost model.

        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available. Train XGBoost model first.")
            return pd.DataFrame()
        return self.feature_importance
