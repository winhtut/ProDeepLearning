import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of entries
num_entries = 10000

# Generate timestamps at hourly intervals
start_datetime = datetime(2022, 11, 10)
timestamps = [start_datetime + timedelta(hours=i) for i in range(num_entries)]

# Initialize random seed for reproducibility
np.random.seed(42)

# Generate synthetic temperature data (°C)
temperature_max = np.random.normal(loc=32, scale=3, size=num_entries)
temperature_min = temperature_max - np.random.uniform(5, 8, size=num_entries)

# Generate synthetic humidity data (%)
humidity_max = np.random.uniform(80, 95, size=num_entries)
humidity_min = humidity_max - np.random.uniform(5, 15, size=num_entries)

# Generate synthetic air pressure data (hPa)
air_pressure_max = np.random.uniform(1005, 1015, size=num_entries)
air_pressure_min = air_pressure_max - np.random.uniform(2, 5, size=num_entries)

# Create the DataFrame
data = pd.DataFrame({
    'Timestamp': timestamps,
    'TemperatureMax': np.round(temperature_max, 1),
    'TemperatureMin': np.round(temperature_min, 1),
    'HumidityMax': np.round(humidity_max, 1),
    'HumidityMin': np.round(humidity_min, 1),
    'AirPressureMax': np.round(air_pressure_max, 1),
    'AirPressureMin': np.round(air_pressure_min, 1)
})

# Save to CSV (optional)
data.to_csv('thai_weather.csv', index=False)

# Display first 5 entries
print(data.head())
