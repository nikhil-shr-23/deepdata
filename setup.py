import subprocess
import sys

def install_packages():
    """Install required packages for the AQI Analysis Dashboard."""
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "plotly",
        "openpyxl"
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")
    print("You can now run the app with: streamlit run app.py")

if __name__ == "__main__":
    install_packages()
