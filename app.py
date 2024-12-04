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

class DashboardApp:
    """
    Main dashboard application class for retail forecasting.
    This class handles the Streamlit interface and coordinates
    between data processing and modeling components.
    """
    def __init__(self):
        """Initialize dashboard components and load necessary data."""
        self.preprocessor = RetailPreprocessor()
        self.forecaster = RetailForecaster()
        self.data_loader = DataLoader("data/")
        
    # In app.py
    def load_data(self):
    """Load and cache data for the dashboard."""
        try:
        features_df, target_df = self.data_loader.load_processed_data()
            return features_df, target_df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None

    def create_sidebar(self):
        """Create the sidebar with control options."""
        st.sidebar.header("Forecast Settings")
        
        # Date range selector
        st.sidebar.subheader("Select Date Range")
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365)
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now()
        )
        
        # Forecast horizon selector
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=90,
            value=30
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        prophet_weight = st.sidebar.slider(
            "Prophet Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4
        )
        
        return start_date, end_date, forecast_horizon, prophet_weight

    def show_data_overview(self, df: pd.DataFrame):
        """Display data overview section."""
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Sales",
                f"${df['TotalAmount'].sum():,.2f}",
                "Historical Data"
            )
            
        with col2:
            st.metric(
                "Average Daily Sales",
                f"${df.groupby('InvoiceDate')['TotalAmount'].sum().mean():,.2f}",
                "Per Day"
            )
            
        with col3:
            st.metric(
                "Total Transactions",
                f"{len(df):,}",
                "Orders"
            )

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
