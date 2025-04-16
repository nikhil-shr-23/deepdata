# Data visual - https://deepdataa.streamlit.app/

# AQI Analysis Dashboard

This Streamlit application provides analysis and visualization of Air Quality Index (AQI) data for Sohna, 2023.

## Features

- **Data Overview**: Basic statistics, time series visualization, and monthly averages
- **Exploratory Analysis**: Detailed analysis of AQI patterns by hour, day, month, and correlations
- **Model Performance**: XGBoost model for AQI prediction with performance metrics and visualizations

## Installation

1. Clone this repository
2. Install the required packages using one of these methods:

   **Option 1:** Using requirements.txt
   ```
   pip install -r requirements.txt
   ```

   **Option 2:** Using setup.py (recommended)
   ```
   python setup.py
   ```

   **Option 3:** Install packages individually
   ```
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost plotly openpyxl
   ```

## Usage

1. Place your AQI Excel file in the same directory as the app (or prepare to upload it)
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Access the app in your web browser at the URL provided in the terminal

### macOS Users

If you encounter an error about `libomp.dylib` when running the app, XGBoost needs the OpenMP library. Fix it by running:

```
./fix_xgboost_macos.sh
```

Or manually install it with:

```
brew install libomp
```

## Data Format

The expected Excel file format has:
- A 'Days' column (numbered 1 to 365)
- Hourly columns (00:00:00, 01:00:00, ..., 23:00:00)
- AQI values in the cells

## Model Details

The app uses XGBoost regression to predict AQI values based on:
- Time features (hour, day of week, month, weekend indicator)
- Lag features (1, 3, 6, and 24 hours)
- Rolling mean features (6 and 24 hours)

Performance metrics include RMSE, MAE, and R².
