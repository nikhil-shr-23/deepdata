@echo off
echo Installing required packages...
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost plotly openpyxl

echo Starting the Streamlit app...
streamlit run app.py
pause
