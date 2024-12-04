# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.preprocessing import RetailPreprocessor
from src.modeling import RetailForecaster
from src.utils import setup_logging, DataLoader

# Set up page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logger = setup_logging()

# app.py

class DashboardApp:
    """Main dashboard application class for retail forecasting."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.preprocessor = RetailPreprocessor()
        self.forecaster = RetailForecaster()
        self.data_loader = DataLoader("data")  # Specify the data directory
        
    def load_data(self):
        """Load and cache data for the dashboard."""
        @st.cache_data
        def load_cached_data():
            try:
                # Try loading the data file
                df = self.data_loader.load_data("retail_data.csv")
                return df
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None
        return load_cached_data()

    def show_data_overview(self, df: pd.DataFrame):
        """
        Display data overview section with proper error handling.
        
        Args:
            df: DataFrame containing the retail data
        """
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

    def plot_sales_trend(self, df: pd.DataFrame):
        """Create and display sales trend visualization."""
        st.subheader("Historical Sales Trend")
        
        daily_sales = df.groupby('InvoiceDate')['TotalAmount'].sum().reset_index()
        
        fig = px.line(
            daily_sales,
            x='InvoiceDate',
            y='TotalAmount',
            title='Daily Sales Over Time'
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_forecast_results(self, predictions: pd.DataFrame):
        """Display forecast results and metrics."""
        st.header("Forecast Results")
        
        # Plot actual vs predicted
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['prophet_pred'],
            name='Prophet Forecast',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['xgb_pred'],
            name='XGBoost Forecast',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['hybrid_pred'],
            name='Hybrid Forecast',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main method to run the dashboard."""
        st.title("Retail Sales Forecasting Dashboard")
        
        # Load data
        df = self.load_data()
        if df is None:
            return
            
        # Get sidebar inputs
        start_date, end_date, horizon, prophet_weight = self.create_sidebar()
        
        # Show data overview
        self.show_data_overview(df)
        
        # Show sales trend
        self.plot_sales_trend(df)
        
        # Generate and show forecast
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Preprocess data
                    processed_data = self.preprocessor.preprocess_data(df)
                    
                    # Train models
                    self.forecaster.train_prophet(
                        processed_data,
                        'InvoiceDate',
                        'TotalAmount'
                    )
                    
                    feature_data = processed_data.drop(
                        ['InvoiceDate', 'TotalAmount'],
                        axis=1
                    )
                    self.forecaster.train_xgboost(
                        feature_data,
                        processed_data['TotalAmount']
                    )
                    
                    # Generate predictions
                    future_dates = pd.DataFrame({
                        'ds': pd.date_range(
                            start=end_date,
                            periods=horizon,
                            freq='D'
                        )
                    })
                    
                    predictions = self.forecaster.make_predictions(
                        future_dates,
                        feature_data,
                        prophet_weight
                    )
                    
                    # Show results
                    self.show_forecast_results(predictions)
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run()
