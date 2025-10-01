"""
Data loading module for homework1.
Loads data from the /database directory.
"""

import os
import numpy as np
import pandas as pd


class DataLoader:
    """Class to handle data loading from various formats."""
    
    def __init__(self, database_path="/database"):
        """
        Initialize the DataLoader.
        
        Parameters:
        -----------
        database_path : str
            Path to the database directory
        """
        self.database_path = database_path
        
    def load_data(self, filename):
        """
        Load data from a file in the database directory.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load
            
        Returns:
        --------
        pd.DataFrame
            Loaded data as a pandas DataFrame
        """
        filepath = os.path.join(self.database_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try to load based on file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(filepath)
        elif file_ext == '.txt':
            # Try different separators
            try:
                return pd.read_csv(filepath, sep=',')
            except:
                try:
                    return pd.read_csv(filepath, sep='\t')
                except:
                    return pd.read_csv(filepath, sep=r'\s+')
        elif file_ext == '.data':
            # Try comma-separated first
            try:
                return pd.read_csv(filepath, header=None)
            except:
                return pd.read_csv(filepath, sep=r'\s+', header=None)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def extract_features_and_labels(self, data, label_column=-1):
        """
        Extract features and labels from the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset
        label_column : int or str
            Index or name of the label column (default: -1, last column)
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
        feature_names : list
            Names of the features
        """
        if isinstance(label_column, int):
            if label_column == -1:
                label_column = data.columns[-1]
            else:
                label_column = data.columns[label_column]
        
        y = data[label_column].values
        X = data.drop(columns=[label_column]).values
        feature_names = [col for col in data.columns if col != label_column]
        
        return X, y, feature_names
    
    def get_dataset_info(self, X, y):
        """
        Get basic information about the dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
            
        Returns:
        --------
        dict
            Dictionary with dataset information
        """
        N = X.shape[0]  # Number of observations
        D = X.shape[1]  # Number of predictors
        classes, counts = np.unique(y, return_counts=True)
        L = len(classes)  # Number of classes
        
        return {
            'N': N,
            'D': D,
            'L': L,
            'classes': classes,
            'class_counts': dict(zip(classes, counts))
        }
