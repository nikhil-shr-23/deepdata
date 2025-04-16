#!/bin/bash

echo "==== XGBoost macOS Fix Script ===="
echo "This script will install the OpenMP library required by XGBoost on macOS."
echo

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is only for macOS. You appear to be running on $(uname)."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Check if Homebrew installation was successful
    if ! command -v brew &> /dev/null; then
        echo "Error: Failed to install Homebrew. Please install it manually."
        echo "Visit https://brew.sh for instructions."
        exit 1
    fi
fi

# Install libomp
echo "Installing libomp using Homebrew..."
brew install libomp

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo
    echo "Success! The OpenMP library has been installed."
    echo "You should now be able to use XGBoost in the AQI Analysis Dashboard."
    echo "Run the app with: streamlit run app.py"
else
    echo
    echo "Error: Failed to install libomp. Please try installing it manually:"
    echo "brew install libomp"
    exit 1
fi
