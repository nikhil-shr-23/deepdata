#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip3 install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost plotly openpyxl

# Check if running on macOS and install libomp if needed
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS. Checking for Homebrew and libomp..."
    if command -v brew &> /dev/null; then
        echo "Homebrew is installed. Installing libomp for XGBoost..."
        brew install libomp
    else
        echo "Homebrew is not installed. XGBoost may not work correctly."
        echo "To install Homebrew and libomp, run:"
        echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "brew install libomp"
    fi
fi

# Run the app
echo "Starting the Streamlit app..."
streamlit run app.py
