"""
Statistical analysis module for homework1.
Implements mono-variate and bi-variate statistical analysis.
"""

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Class to perform statistical analysis on data."""
    
    @staticmethod
    def calculate_statistics(data):
        """
        Calculate mean, standard deviation, and skewness for data.
        
        Parameters:
        -----------
        data : np.ndarray
            1D array of data
            
        Returns:
        --------
        dict
            Dictionary with mean, std, and skewness
        """
        return {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),  # Sample standard deviation
            'skewness': stats.skew(data)
        }
    
    @staticmethod
    def unconditional_monovariate_analysis(X, feature_names):
        """
        Perform unconditional mono-variate analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        feature_names : list
            Names of the features
            
        Returns:
        --------
        dict
            Dictionary with statistics for each feature
        """
        D = X.shape[1]
        results = {}
        
        for d in range(D):
            feature_name = feature_names[d] if d < len(feature_names) else f"Feature_{d}"
            results[feature_name] = StatisticalAnalyzer.calculate_statistics(X[:, d])
        
        return results
    
    @staticmethod
    def conditional_monovariate_analysis(X, y, feature_names):
        """
        Perform class-conditional mono-variate analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
        feature_names : list
            Names of the features
            
        Returns:
        --------
        dict
            Nested dictionary with statistics for each feature and class
        """
        D = X.shape[1]
        classes = np.unique(y)
        results = {}
        
        for d in range(D):
            feature_name = feature_names[d] if d < len(feature_names) else f"Feature_{d}"
            results[feature_name] = {}
            
            for cls in classes:
                mask = y == cls
                results[feature_name][cls] = StatisticalAnalyzer.calculate_statistics(X[mask, d])
        
        return results
    
    @staticmethod
    def calculate_correlation_matrix(X):
        """
        Calculate the correlation matrix for all pairs of features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
            
        Returns:
        --------
        np.ndarray
            Correlation matrix (D x D)
        """
        return np.corrcoef(X.T)
    
    @staticmethod
    def format_statistics_table(stats_dict, title="Statistics"):
        """
        Format statistics dictionary as a readable table string.
        
        Parameters:
        -----------
        stats_dict : dict
            Dictionary with statistics
        title : str
            Title for the table
            
        Returns:
        --------
        str
            Formatted table string
        """
        lines = [f"\n{title}", "=" * 80]
        lines.append(f"{'Feature':<20} {'Mean':>12} {'Std Dev':>12} {'Skewness':>12}")
        lines.append("-" * 80)
        
        for feature, stats in stats_dict.items():
            lines.append(
                f"{feature:<20} {stats['mean']:>12.4f} {stats['std']:>12.4f} "
                f"{stats['skewness']:>12.4f}"
            )
        
        return "\n".join(lines)
    
    @staticmethod
    def format_conditional_statistics_table(stats_dict, title="Class-Conditional Statistics"):
        """
        Format class-conditional statistics dictionary as a readable table string.
        
        Parameters:
        -----------
        stats_dict : dict
            Nested dictionary with statistics for each feature and class
        title : str
            Title for the table
            
        Returns:
        --------
        str
            Formatted table string
        """
        lines = [f"\n{title}", "=" * 100]
        lines.append(
            f"{'Feature':<20} {'Class':>8} {'Mean':>12} {'Std Dev':>12} {'Skewness':>12}"
        )
        lines.append("-" * 100)
        
        for feature, class_stats in stats_dict.items():
            for i, (cls, stats) in enumerate(class_stats.items()):
                feature_label = feature if i == 0 else ""
                lines.append(
                    f"{feature_label:<20} {str(cls):>8} {stats['mean']:>12.4f} "
                    f"{stats['std']:>12.4f} {stats['skewness']:>12.4f}"
                )
        
        return "\n".join(lines)
