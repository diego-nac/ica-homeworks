"""
Data loading utilities for the ICA homeworks project.
"""

import pandas as pd
from typing import Optional, Union
from pathlib import Path


class DataLoader:
    """
    A class to handle data loading operations for the ML project.
    """
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = Path(data_path) if data_path else None
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
        
        Returns:
            DataFrame containing the loaded data
        """
        if self.data_path:
            filepath = self.data_path / filename
        else:
            filepath = Path(filename)
        
        return pd.read_csv(filepath, **kwargs)
    
    def load_excel(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            filename: Name of the Excel file
            **kwargs: Additional arguments to pass to pd.read_excel
        
        Returns:
            DataFrame containing the loaded data
        """
        if self.data_path:
            filepath = self.data_path / filename
        else:
            filepath = Path(filename)
        
        return pd.read_excel(filepath, **kwargs)
    
    def save_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the output CSV file
            **kwargs: Additional arguments to pass to df.to_csv
        """
        if self.data_path:
            filepath = self.data_path / filename
        else:
            filepath = Path(filename)
        
        df.to_csv(filepath, index=False, **kwargs)
