import pandas as pd
# This script checks the columns used for Water Quality Index (WQI) calculation in the dataset.
# Columns used for WQI calculation
wqi_cols = [
    'pH Level',
    'Dissolved Oxygen (mg/L)',
    'Nitrate-N/Nitrite-N  (mg/L)',
    'Ammonia (mg/L)',
    'Phosphate (mg/L)',
    'Surface Water Temp (°C)',
    'Middle Water Temp (°C)',
    'Bottom Water Temp (°C)'
]

df = pd.read_csv('processed data/processed_data.csv')

print('Non-null counts for WQI columns:')
print(df[wqi_cols].count())
print('\nData types for WQI columns:')
print(df[wqi_cols].dtypes)
print('\nSample values for WQI columns:')
print(df[wqi_cols].head(10)) 