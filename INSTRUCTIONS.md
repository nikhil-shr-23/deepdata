# AQI Analysis Dashboard - Instructions

## Quick Start

### For macOS/Linux:
```bash
# Make the script executable (one-time setup)
chmod +x run_app.sh

# Run the app
./run_app.sh
```

### For Windows:
```
# Double-click run_app.bat
# OR
# Run from command prompt
run_app.bat
```

## Manual Installation

If the quick start scripts don't work, follow these steps:

1. Install the required packages:
   ```
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost plotly openpyxl
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Troubleshooting

### XGBoost Installation Issues

#### macOS OpenMP Error

If you see an error like this on macOS:
```
XGBoost Library (libxgboost.dylib) could not be loaded.
Library not loaded: @rpath/libomp.dylib
```

This is because XGBoost requires the OpenMP runtime library. To fix it:

1. Install Homebrew if you don't have it:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install libomp:
   ```
   brew install libomp
   ```

3. Restart the app

#### Other XGBoost Issues

If you encounter other issues with XGBoost installation:

1. Make sure you have a C++ compiler installed:
   - Windows: Install Visual C++ Build Tools
   - macOS: Install Xcode Command Line Tools (`xcode-select --install`)
   - Linux: Install GCC (`sudo apt-get install build-essential` for Ubuntu)

2. Try installing XGBoost separately:
   ```
   pip install xgboost
   ```

3. If you still have issues, you can use the app without the Model Performance section. The app will automatically show alternative visualizations.

### Data File Issues

Make sure your Excel file has the following format:
- A 'Days' column (numbered 1 to 365)
- Hourly columns (00:00:00, 01:00:00, ..., 23:00:00)
- AQI values in the cells

A sample file named `AQI_hourly_city_level_sohna_2023.xlsx` should be in the same directory as the app.
