"""
Helper functions
"""

import pandas as pd

def get_wqi_classification(wqi_value):
    """Get WQI classification and advice"""
    if wqi_value < 50:
        return "Excellent", "Water is suitable for all uses. Maintain current practices."
    elif 50 <= wqi_value < 75:
        return "Good", "Water is suitable for most uses. Minor treatment might be needed. Some monitoring is recommended."
    elif 75 <= wqi_value < 90:
        return "Fair", "Water is suitable for limited uses. Aquatic life might be stressed. Further treatment may be needed."
    else:
        return "Poor", "Water is not suitable for any uses without treatment."

def generate_future_dates(start_date, num_periods):
    """Generate future dates"""
    return pd.date_range(start=start_date, periods=num_periods, freq='M')

def create_forecast_dataframe(predictions):
    """Create forecast dataframe with all outputs"""
    columns = [
        'WQI',
        'Ammonia (mg/L)',
        'Nitrate-N/Nitrite-N  (mg/L)',
        'Phosphate (mg/L)'
    ]
    return pd.DataFrame([predictions], columns=columns)