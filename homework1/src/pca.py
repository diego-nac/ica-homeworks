"""
Principal Component Analysis (PCA) module for homework1.
Custom implementation without using pre-made PCA functions.
"""

import numpy as np


class PCA:
    """Custom implementation of Principal Component Analysis."""
    
    def __init__(self, n_components=2):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components to retain
        """
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.principal_components = None
        
    def fit(self, X):
        """
        Fit PCA on the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        """
        # Step 1: Standardize the data (mean=0, std=1)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        
        X_standardized = (X - self.mean) / self.std
        
        # Step 2: Calculate the covariance matrix
        N = X_standardized.shape[0]
        covariance_matrix = (X_standardized.T @ X_standardized) / (N - 1)
        
        # Step 3: Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select the top n_components eigenvectors
        self.principal_components = self.eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data to the principal component space.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
            
        Returns:
        --------
        np.ndarray
            Projected data (N x n_components)
        """
        if self.principal_components is None:
            raise ValueError("PCA must be fitted before transforming data")
        
        # Standardize the data using the same mean and std from fitting
        X_standardized = (X - self.mean) / self.std
        
        # Project onto principal components
        X_projected = X_standardized @ self.principal_components
        
        return X_projected
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
            
        Returns:
        --------
        np.ndarray
            Projected data (N x n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def explained_variance_ratio(self):
        """
        Calculate the proportion of variance explained by each principal component.
        
        Returns:
        --------
        np.ndarray
            Proportion of variance explained by each component
        """
        if self.eigenvalues is None:
            raise ValueError("PCA must be fitted before calculating explained variance")
        
        total_variance = np.sum(self.eigenvalues)
        return self.eigenvalues[:self.n_components] / total_variance
    
    def get_loadings(self):
        """
        Get the loadings (principal components) matrix.
        
        Returns:
        --------
        np.ndarray
            Loadings matrix (D x n_components)
        """
        if self.principal_components is None:
            raise ValueError("PCA must be fitted before getting loadings")
        
        return self.principal_components
    
    def summary(self):
        """
        Print a summary of the PCA results.
        """
        if self.eigenvalues is None:
            print("PCA has not been fitted yet.")
            return
        
        print("\nPCA Summary")
        print("=" * 60)
        print(f"Number of components: {self.n_components}")
        print(f"\nEigenvalues (all): {self.eigenvalues}")
        print(f"\nExplained variance ratio:")
        
        explained_var = self.explained_variance_ratio()
        cumulative_var = np.cumsum(explained_var)
        
        for i in range(self.n_components):
            print(f"  PC{i+1}: {explained_var[i]:.4f} ({explained_var[i]*100:.2f}%) "
                  f"- Cumulative: {cumulative_var[i]:.4f} ({cumulative_var[i]*100:.2f}%)")
        
        print("\nPrincipal Components (Loadings):")
        print(self.principal_components)
        print("=" * 60)
