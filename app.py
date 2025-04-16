import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    st.error("""
    XGBoost is not installed. Please install it using:
    ```
    pip install xgboost
    ```
    """)
except Exception as e:
    XGBOOST_AVAILABLE = False
    error_msg = str(e)
    if "libomp.dylib" in error_msg and "Mac OSX" in error_msg:
        st.error("""
        ### XGBoost OpenMP Error on macOS

        XGBoost requires the OpenMP runtime library which is missing on your system.

        To fix this, open Terminal and run:
        ```
        brew install libomp
        ```

        If you don't have Homebrew installed, install it first with:
        ```
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```

        After installing libomp, restart this app.
        """)
    else:
        st.error(f"Error loading XGBoost: {e}")

# Set page configuration
st.set_page_config(
    page_title="AQI Analysis Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# Title and description
st.title("Air Quality Index (AQI) Analysis Dashboard")
st.markdown("""
This dashboard analyzes hourly Air Quality Index (AQI) data for Sohna, 2023.
Upload your own AQI data or use the sample data provided.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Performance"])

# Function to load and process data
@st.cache_data
def load_and_process_data(file):
    try:
        # Load the Excel file
        df_raw = pd.read_excel(file)

        # Melt the dataframe to long format
        df = df_raw.melt(id_vars='Days', var_name='Hour', value_name='AQI')

        # Convert 'Days' to actual date assuming Day 1 = 2023-01-01
        df['Date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['Days'] - 1, unit='D')

        # Combine date and hour into full datetime
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hour'].astype(str))

        # Drop now-unnecessary columns
        df.drop(['Days', 'Hour', 'Date'], axis=1, inplace=True)

        # Sort by datetime
        df.sort_values('Datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create features
        df['hour'] = df['Datetime'].dt.hour
        df['dayofweek'] = df['Datetime'].dt.dayofweek
        df['month'] = df['Datetime'].dt.month
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

        # Create lag features
        for lag in [1, 3, 6, 24]:
            df[f'AQI_lag_{lag}'] = df['AQI'].shift(lag)

        # Create rolling mean features
        df['AQI_roll_mean_6'] = df['AQI'].rolling(window=6).mean()
        df['AQI_roll_mean_24'] = df['AQI'].rolling(window=24).mean()

        # Drop rows with NaN values
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Categorize AQI values
        def categorize_aqi(value):
            if value <= 50:
                return 'Good'
            elif value <= 100:
                return 'Satisfactory'
            elif value <= 200:
                return 'Moderate'
            elif value <= 300:
                return 'Poor'
            elif value <= 400:
                return 'Very Poor'
            else:
                return 'Severe'

        df['AQI_Class'] = df['AQI'].apply(categorize_aqi)

        return df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# Function to train model
@st.cache_resource
def train_model(df):
    # Check if XGBoost is available
    if not XGBOOST_AVAILABLE:
        st.error("XGBoost is required for model training. Please install it and restart the app.")
        return None

    # Prepare features and target
    features = ['hour', 'dayofweek', 'month', 'is_weekend',
                'AQI_lag_1', 'AQI_lag_3', 'AQI_lag_6', 'AQI_lag_24',
                'AQI_roll_mean_6', 'AQI_roll_mean_24']

    X = df[features]
    y = df['AQI']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Results dictionary
    results = {
        'model': model,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'feature_importance': feature_importance
    }

    return results

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload AQI Excel file", type=["xlsx"])

# Use sample data if no file is uploaded
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
else:
    # Try to load the sample file if it exists
    try:
        df = load_and_process_data("AQI_hourly_city_level_sohna_2023.xlsx")
        st.sidebar.success("Using sample data. Upload your own file to analyze different data.")
    except:
        st.sidebar.warning("Sample data not found. Please upload an Excel file.")
        df = None

# Only proceed if data is loaded
if df is not None:
    # Train model if XGBoost is available
    model_results = train_model(df) if XGBOOST_AVAILABLE else None

    # DATA OVERVIEW PAGE
    if page == "Data Overview":
        st.header("Data Overview")

        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Time Range:** {df['Datetime'].min().date()} to {df['Datetime'].max().date()}")
            st.write(f"**Number of Records:** {len(df)}")
            st.write(f"**Average AQI:** {df['AQI'].mean():.2f}")
            st.write(f"**Maximum AQI:** {df['AQI'].max():.2f}")
            st.write(f"**Minimum AQI:** {df['AQI'].min():.2f}")

        with col2:
            st.subheader("AQI Class Distribution")
            aqi_class_counts = df['AQI_Class'].value_counts()
            fig = px.pie(
                values=aqi_class_counts.values,
                names=aqi_class_counts.index,
                title="Distribution of AQI Classes",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig)

        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))

        # Time series plot
        st.subheader("AQI Time Series")
        fig = px.line(
            df, x='Datetime', y='AQI',
            title='Air Quality Index Over Time',
            labels={'AQI': 'Air Quality Index', 'Datetime': 'Date & Time'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monthly average
        st.subheader("Monthly Average AQI")
        monthly_avg = df.groupby(df['Datetime'].dt.month)['AQI'].mean().reset_index()
        monthly_avg['Month'] = monthly_avg['Datetime'].apply(lambda x: datetime(2023, x, 1).strftime('%B'))
        fig = px.bar(
            monthly_avg, x='Month', y='AQI',
            title='Average AQI by Month',
            labels={'AQI': 'Average Air Quality Index', 'Month': 'Month'},
            color='AQI',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # EXPLORATORY ANALYSIS PAGE
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")

        # AQI by hour of day
        st.subheader("AQI by Hour of Day")
        hourly_avg = df.groupby('hour')['AQI'].mean().reset_index()
        fig = px.line(
            hourly_avg, x='hour', y='AQI',
            title='Average AQI by Hour of Day',
            labels={'AQI': 'Average Air Quality Index', 'hour': 'Hour of Day'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # AQI by day of week
        st.subheader("AQI by Day of Week")
        day_avg = df.groupby('dayofweek')['AQI'].mean().reset_index()
        day_avg['Day'] = day_avg['dayofweek'].apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        fig = px.bar(
            day_avg, x='Day', y='AQI',
            title='Average AQI by Day of Week',
            labels={'AQI': 'Average Air Quality Index', 'Day': 'Day of Week'},
            color='AQI',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: Hour vs Month
        st.subheader("AQI Heatmap: Hour vs Month")
        pivot_df = df.pivot_table(
            index='hour',
            columns=df['Datetime'].dt.month,
            values='AQI',
            aggfunc='mean'
        )

        # Rename columns to month names
        pivot_df.columns = [datetime(2023, month, 1).strftime('%B') for month in pivot_df.columns]

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Month", y="Hour of Day", color="Average AQI"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='Viridis',
            title="Average AQI by Hour and Month"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution of AQI values
        st.subheader("Distribution of AQI Values")
        fig = px.histogram(
            df, x='AQI',
            nbins=50,
            title='Distribution of AQI Values',
            labels={'AQI': 'Air Quality Index'},
            color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        numeric_cols = ['AQI', 'hour', 'dayofweek', 'month', 'is_weekend',
                        'AQI_lag_1', 'AQI_lag_3', 'AQI_lag_6', 'AQI_lag_24',
                        'AQI_roll_mean_6', 'AQI_roll_mean_24']
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Features",
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig, use_container_width=True)

    # MODEL PERFORMANCE PAGE
    elif page == "Model Performance":
        st.header("Model Performance")

        if not XGBOOST_AVAILABLE:
            st.error("XGBoost is required for the Model Performance section.")

            # Provide alternative visualization without XGBoost
            st.subheader("Alternative Analysis: Feature Correlations")
            st.write("""
            Since XGBoost is not available, we're showing feature correlations instead of model performance.
            This can help identify which features are most strongly related to AQI values.
            """)

            # Create correlation with AQI
            numeric_cols = ['hour', 'dayofweek', 'month', 'is_weekend',
                           'AQI_lag_1', 'AQI_lag_3', 'AQI_lag_6', 'AQI_lag_24',
                           'AQI_roll_mean_6', 'AQI_roll_mean_24']

            corr_with_aqi = pd.DataFrame({
                'Feature': numeric_cols,
                'Correlation': [df['AQI'].corr(df[col]) for col in numeric_cols]
            }).sort_values('Correlation', ascending=False)

            fig = px.bar(
                corr_with_aqi,
                x='Correlation', y='Feature',
                orientation='h',
                title='Feature Correlation with AQI',
                labels={'Correlation': 'Correlation with AQI', 'Feature': 'Feature'},
                color='Correlation',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            ### How to Fix XGBoost Issues on macOS

            If you're on macOS and seeing an error about 'libomp.dylib', you need to install the OpenMP library:

            1. Open Terminal
            2. Run: `brew install libomp`
            3. Restart this app

            If you don't have Homebrew installed, install it first with:
            ```
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            ```
            """)

            st.stop()

        if model_results is None:
            st.error("Model training failed. Please check the logs for more information.")
            st.stop()

        # Model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test RMSE", f"{model_results['test_rmse']:.2f}")
        with col2:
            st.metric("Test MAE", f"{model_results['test_mae']:.2f}")
        with col3:
            st.metric("Test R²", f"{model_results['test_r2']:.4f}")

        # Training vs Test metrics
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²'],
            'Training': [model_results['train_rmse'], model_results['train_mae'], model_results['train_r2']],
            'Testing': [model_results['test_rmse'], model_results['test_mae'], model_results['test_r2']]
        })

        st.subheader("Training vs Testing Metrics")
        fig = px.bar(
            metrics_df, x='Metric', y=['Training', 'Testing'],
            barmode='group',
            title='Model Performance Metrics',
            labels={'value': 'Value', 'variable': 'Dataset'},
            color_discrete_sequence=['#3366CC', '#DC3912']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance")
        fig = px.bar(
            model_results['feature_importance'],
            x='Importance', y='Feature',
            orientation='h',
            title='XGBoost Feature Importance',
            labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Actual vs Predicted
        st.subheader("Actual vs Predicted Values")

        # Create a dataframe for plotting
        pred_df = pd.DataFrame({
            'Actual': model_results['y_test'],
            'Predicted': model_results['y_pred_test']
        })

        fig = px.scatter(
            pred_df, x='Actual', y='Predicted',
            title='Actual vs Predicted AQI Values (Test Set)',
            labels={'Actual': 'Actual AQI', 'Predicted': 'Predicted AQI'},
            opacity=0.6
        )

        # Add perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Residuals plot
        st.subheader("Residuals Analysis")

        pred_df['Residuals'] = pred_df['Actual'] - pred_df['Predicted']

        fig = px.scatter(
            pred_df, x='Predicted', y='Residuals',
            title='Residuals vs Predicted Values',
            labels={'Predicted': 'Predicted AQI', 'Residuals': 'Residuals (Actual - Predicted)'},
            opacity=0.6
        )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")

        st.plotly_chart(fig, use_container_width=True)

        # Residuals distribution
        fig = px.histogram(
            pred_df, x='Residuals',
            nbins=50,
            title='Distribution of Residuals',
            labels={'Residuals': 'Residuals (Actual - Predicted)'},
            color_discrete_sequence=['#3366CC']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Time-based prediction (if datetime is available in test set)
        st.subheader("Predictions Over Time")
        st.write("Note: This is a sample of predictions over time and may not represent the actual time sequence due to random train-test split.")

        # Get a sample of predictions for visualization
        sample_size = min(100, len(model_results['y_test']))
        sample_indices = np.random.choice(len(model_results['y_test']), sample_size, replace=False)

        time_df = pd.DataFrame({
            'Index': sample_indices,
            'Actual': model_results['y_test'].iloc[sample_indices].values,
            'Predicted': model_results['y_pred_test'][sample_indices]
        }).sort_values('Index')

        fig = px.line(
            time_df, x='Index', y=['Actual', 'Predicted'],
            title='Actual vs Predicted AQI Values Over Samples',
            labels={'value': 'AQI', 'Index': 'Sample Index', 'variable': 'Type'},
            color_discrete_map={'Actual': '#3366CC', 'Predicted': '#DC3912'}
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload an AQI Excel file to begin analysis.")
