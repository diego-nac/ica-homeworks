"""
Visualization module for homework1.
Creates histograms, box-plots, scatter plots, and other visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """Class to create visualizations for data analysis."""
    
    def __init__(self, output_dir="outputs/figures"):
        """
        Initialize the Visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save figures
        """
        self.output_dir = output_dir
        sns.set_style("whitegrid")
        
    def plot_unconditional_histograms(self, X, feature_names, save=True):
        """
        Plot unconditional histograms for all features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        D = X.shape[1]
        n_cols = min(3, D)
        n_rows = (D + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if D == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for d in range(D):
            feature_name = feature_names[d] if d < len(feature_names) else f"Feature_{d}"
            axes[d].hist(X[:, d], bins=30, edgecolor='black', alpha=0.7)
            axes[d].set_xlabel(feature_name)
            axes[d].set_ylabel('Frequency')
            axes[d].set_title(f'Histogram of {feature_name}')
            axes[d].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for d in range(D, len(axes)):
            axes[d].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/unconditional_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_unconditional_boxplots(self, X, feature_names, save=True):
        """
        Plot unconditional box-plots for all features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        D = X.shape[1]
        
        fig, ax = plt.subplots(figsize=(max(10, D*1.5), 6))
        
        positions = np.arange(D)
        bp = ax.boxplot([X[:, d] for d in range(D)], positions=positions, 
                        patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xticks(positions)
        ax.set_xticklabels([feature_names[d] if d < len(feature_names) else f"F{d}" 
                           for d in range(D)], rotation=45, ha='right')
        ax.set_ylabel('Values')
        ax.set_title('Box-plots of All Features')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/unconditional_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_histograms(self, X, y, feature_names, save=True):
        """
        Plot class-conditional histograms for all features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        D = X.shape[1]
        classes = np.unique(y)
        L = len(classes)
        
        n_cols = min(3, D)
        n_rows = (D + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if D == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, L))
        
        for d in range(D):
            feature_name = feature_names[d] if d < len(feature_names) else f"Feature_{d}"
            
            for i, cls in enumerate(classes):
                mask = y == cls
                axes[d].hist(X[mask, d], bins=20, alpha=0.6, label=f'Class {cls}', 
                           color=colors[i], edgecolor='black')
            
            axes[d].set_xlabel(feature_name)
            axes[d].set_ylabel('Frequency')
            axes[d].set_title(f'Class-Conditional Histogram of {feature_name}')
            axes[d].legend()
            axes[d].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for d in range(D, len(axes)):
            axes[d].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/conditional_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_conditional_boxplots(self, X, y, feature_names, save=True):
        """
        Plot class-conditional box-plots for all features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        D = X.shape[1]
        classes = np.unique(y)
        L = len(classes)
        
        n_cols = min(2, D)
        n_rows = (D + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
        if D == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, L))
        
        for d in range(D):
            feature_name = feature_names[d] if d < len(feature_names) else f"Feature_{d}"
            
            data_by_class = [X[y == cls, d] for cls in classes]
            bp = axes[d].boxplot(data_by_class, labels=[f'Class {cls}' for cls in classes],
                               patch_artist=True, widths=0.6)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[d].set_xlabel('Class')
            axes[d].set_ylabel('Values')
            axes[d].set_title(f'Class-Conditional Box-plot of {feature_name}')
            axes[d].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for d in range(D, len(axes)):
            axes[d].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/conditional_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scatter_matrix(self, X, y, feature_names, save=True):
        """
        Plot scatter plots for all pairs of features with class colors.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (N x D)
        y : np.ndarray
            Label vector (N,)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        D = X.shape[1]
        classes = np.unique(y)
        
        fig, axes = plt.subplots(D, D, figsize=(3*D, 3*D))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        color_map = {cls: colors[i] for i, cls in enumerate(classes)}
        
        for i in range(D):
            for j in range(D):
                ax = axes[i, j] if D > 1 else axes
                
                if i == j:
                    # Diagonal: histograms
                    for cls in classes:
                        mask = y == cls
                        ax.hist(X[mask, i], bins=20, alpha=0.6, 
                               color=color_map[cls], edgecolor='black')
                    ax.set_yticks([])
                else:
                    # Off-diagonal: scatter plots
                    for cls in classes:
                        mask = y == cls
                        ax.scatter(X[mask, j], X[mask, i], c=[color_map[cls]], 
                                 alpha=0.6, s=30, label=f'Class {cls}')
                
                # Labels
                if i == D - 1:
                    ax.set_xlabel(feature_names[j] if j < len(feature_names) else f"F{j}", 
                                fontsize=8)
                else:
                    ax.set_xticks([])
                
                if j == 0:
                    ax.set_ylabel(feature_names[i] if i < len(feature_names) else f"F{i}", 
                                fontsize=8)
                else:
                    ax.set_yticks([])
                
                ax.grid(True, alpha=0.3)
        
        # Add legend
        handles, labels = axes[0, 1].get_legend_handles_labels() if D > 1 else ([], [])
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/scatter_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self, correlation_matrix, feature_names, save=True):
        """
        Plot correlation matrix as a heatmap.
        
        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation matrix (D x D)
        feature_names : list
            Names of the features
        save : bool
            Whether to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pca_projection(self, X_pca, y, save=True):
        """
        Plot PCA projection (first two principal components).
        
        Parameters:
        -----------
        X_pca : np.ndarray
            Projected data (N x 2)
        y : np.ndarray
            Label vector (N,)
        save : bool
            Whether to save the figure
        """
        classes = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, cls in enumerate(classes):
            mask = y == cls
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], 
                      alpha=0.6, s=50, label=f'Class {cls}')
        
        ax.set_xlabel('First Principal Component (PC1)')
        ax.set_ylabel('Second Principal Component (PC2)')
        ax.set_title('PCA Projection (First Two Principal Components)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/pca_projection.png", dpi=300, bbox_inches='tight')
        plt.close()
