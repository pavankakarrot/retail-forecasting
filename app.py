# app.py
# Purpose: Main Streamlit dashboard application

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.preprocessing import RetailPreprocessor
from src.modeling import RetailForecaster
from src.utils import setup_logging, DataLoader

# Set up logging
logger = setup_logging()

# Set up page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DashboardApp:
    """Main dashboard application class for retail forecasting."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.preprocessor = RetailPreprocessor()
        self.forecaster = RetailForecaster()
        self.data_loader = DataLoader("data")
    
    def load_data(self):
        """Load and cache data for the dashboard."""
        @st.cache_data
        def load_cached_data():
            try:
                df = self.data_loader.load_data("retail_data.csv")
                return df
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None
        return load_cached_data()

    def show_data_overview(self, df: pd.DataFrame):
        """Display data overview section."""
        if df is None:
            st.error("No data available to display")
            return
            
        st.header("Data Overview")
        
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_amount = df['TotalAmount'].sum() if 'TotalAmount' in df.columns else 0
                st.metric(
                    "Total Sales",
                    f"${total_amount:,.2f}",
                    "Historical Data"
                )
            
            with col2:
                if 'InvoiceDate' in df.columns and 'TotalAmount' in df.columns:
                    daily_avg = df.groupby('InvoiceDate')['TotalAmount'].sum().mean()
                else:
                    daily_avg = 0
                st.metric(
                    "Average Daily Sales",
                    f"${daily_avg:,.2f}",
                    "Per Day"
                )
            
            with col3:
                st.metric(
                    "Total Transactions",
                    f"{len(df):,}",
                    "Orders"
                )
        
        except Exception as e:
            st.error(f"Error displaying data overview: {str(e)}")
            logger.error(f"Data overview error: {str(e)}")

    def run(self):
        """Main method to run the dashboard."""
        st.title("Retail Sales Forecasting Dashboard")
        
        # Load data
        df = self.load_data()
        if df is None:
            st.error("Failed to load data. Please check data file location and format.")
            return
        
        # Show data overview
        self.show_data_overview(df)

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()
