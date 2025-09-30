# Database Module

## Overview
This module provides utilities for data management and loading operations used across all homeworks.

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

# Load a CSV file
df = loader.load_csv('dataset.csv')

# Save processed data
loader.save_csv(df, 'processed_dataset.csv')
```

## Future Extensions
- Database connection support (SQL, NoSQL)
- Data validation utilities
- Data versioning
- Caching mechanisms
