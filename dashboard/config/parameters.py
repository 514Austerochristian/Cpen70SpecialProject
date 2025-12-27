# dashboard/config/parameters.py
"""
Parameter definitions for water quality monitoring
"""

# Water quality parameters
WATER_PARAMETERS = [
    'Surface Water Temp (°C)', 
    'Middle Water Temp (°C)', 
    'Bottom Water Temp (°C)',
    'pH Level', 
    'Dissolved Oxygen (mg/L)'
]

# Climate parameters
CLIMATE_PARAMETERS = [
    'RAINFALL', 
    'TMAX', 
    'TMIN', 
    'RH', 
    'WIND_SPEED', 
    'WIND_DIRECTION'
]

# Pollutant parameters
POLLUTANT_PARAMETERS = [
    'Ammonia (mg/L)', 
    'Nitrate-N/Nitrite-N  (mg/L)', 
    'Phosphate (mg/L)'
]

# Parameter combinations
PARAMETER_COMBINATIONS = {
    "WQI + Pollutant Level": WATER_PARAMETERS + CLIMATE_PARAMETERS + POLLUTANT_PARAMETERS,
    "WQI Only": WATER_PARAMETERS,
    "WQI + Climate": WATER_PARAMETERS + CLIMATE_PARAMETERS
}

# Full feature columns (must match training order)
FULL_FEATURE_COLUMNS = [
    'Surface Water Temp (°C)',
    'Middle Water Temp (°C)',
    'Bottom Water Temp (°C)',
    'pH Level',
    'Dissolved Oxygen (mg/L)',
    'RAINFALL',
    'TMAX',
    'TMIN',
    'RH',
    'WIND_SPEED',
    'WIND_DIRECTION',
    'Ammonia (mg/L)',
    'Nitrate-N/Nitrite-N  (mg/L)',
    'Phosphate (mg/L)'
]