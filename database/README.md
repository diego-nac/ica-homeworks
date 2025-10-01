# Database Module

## Overview
This module provides utilities for data management and loading operations used across all homeworks. The primary dataset for this project is the **Air Quality Index** dataset from the UCI Machine Learning Repository.

## Dataset Information

### Air Quality Index Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **Instances**: 9358 hourly averaged responses
- **Sensors**: 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device
- **Purpose**: Analyzing air quality measurements and predicting pollutant concentrations

### Dataset Features
The dataset typically includes measurements of:
- CO (Carbon Monoxide)
- NOx (Nitrogen Oxides)
- NO2 (Nitrogen Dioxide)
- O3 (Ozone)
- Temperature
- Relative Humidity
- Absolute Humidity

## Components

### `data_loader.py`
Provides the `DataLoader` class for:
- Loading data from CSV files
- Loading data from Excel files
- Saving processed data to CSV
- Managing data paths

## Usage Example

```python
from database.data_loader import DataLoader

# Initialize the loader
loader = DataLoader(data_path='./data')

# Load the Air Quality dataset
df = loader.load_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Save processed data
loader.save_csv(df, 'processed_air_quality.csv')
```

## Future Extensions
- Database connection support (SQL, NoSQL)
- Data validation utilities
- Data versioning
- Caching mechanisms
