"""
Main analysis script for Homework 1.

This script performs a comprehensive exploratory data analysis including:
1. Dataset description
2. Unconditional mono-variate analysis
3. Class-conditional mono-variate analysis
4. Unconditional bi-variate analysis
5. Unconditional multi-variate analysis (PCA)
"""

import os
import sys
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from statistical_analysis import StatisticalAnalyzer
from visualization import Visualizer
from pca import PCA


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def step1_dataset_description(X, y, feature_names):
    """
    Step 1: Describe the dataset and its features.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    y : np.ndarray
        Label vector (N,)
    feature_names : list
        Names of the features
    """
    print_section_header("STEP 1: DATASET DESCRIPTION")
    
    N = X.shape[0]  # Number of observations
    D = X.shape[1]  # Number of predictors
    classes, counts = np.unique(y, return_counts=True)
    L = len(classes)  # Number of classes
    
    print(f"\nNumber of observations (N): {N}")
    print(f"Number of predictor variables (D): {D}")
    print(f"Number of classes (L): {L}")
    print(f"\nFeature names: {feature_names}")
    
    print(f"\nClass distribution:")
    for cls, count in zip(classes, counts):
        percentage = (count / N) * 100
        print(f"  Class {cls}: {count} observations ({percentage:.2f}%)")


def step2_unconditional_analysis(X, feature_names, visualizer):
    """
    Step 2: Perform unconditional mono-variate analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    feature_names : list
        Names of the features
    visualizer : Visualizer
        Visualizer instance
    """
    print_section_header("STEP 2: UNCONDITIONAL MONO-VARIATE ANALYSIS")
    
    # Calculate statistics
    stats = StatisticalAnalyzer.unconditional_monovariate_analysis(X, feature_names)
    
    # Print statistics table
    print(StatisticalAnalyzer.format_statistics_table(stats, 
          "Unconditional Statistics for All Features"))
    
    # Create visualizations
    print("\nGenerating unconditional histograms...")
    visualizer.plot_unconditional_histograms(X, feature_names)
    
    print("Generating unconditional box-plots...")
    visualizer.plot_unconditional_boxplots(X, feature_names)
    
    print("\n✓ Unconditional analysis complete. Plots saved to outputs/figures/")
    
    return stats


def step3_conditional_analysis(X, y, feature_names, visualizer):
    """
    Step 3: Perform class-conditional mono-variate analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    y : np.ndarray
        Label vector (N,)
    feature_names : list
        Names of the features
    visualizer : Visualizer
        Visualizer instance
    """
    print_section_header("STEP 3: CLASS-CONDITIONAL MONO-VARIATE ANALYSIS")
    
    # Calculate statistics
    stats = StatisticalAnalyzer.conditional_monovariate_analysis(X, y, feature_names)
    
    # Print statistics table
    print(StatisticalAnalyzer.format_conditional_statistics_table(stats,
          "Class-Conditional Statistics"))
    
    # Create visualizations
    print("\nGenerating class-conditional histograms...")
    visualizer.plot_conditional_histograms(X, y, feature_names)
    
    print("Generating class-conditional box-plots...")
    visualizer.plot_conditional_boxplots(X, y, feature_names)
    
    print("\n✓ Class-conditional analysis complete. Plots saved to outputs/figures/")
    
    # Analyze discriminative power
    print("\nDiscriminative Power Analysis:")
    print("-" * 80)
    classes = np.unique(y)
    for feature in feature_names:
        means = [stats[feature][cls]['mean'] for cls in classes]
        stds = [stats[feature][cls]['std'] for cls in classes]
        mean_diff = np.max(means) - np.min(means)
        avg_std = np.mean(stds)
        
        # Simple heuristic: if mean difference > 2 * average std, it has discriminative power
        if mean_diff > 2 * avg_std:
            print(f"  {feature}: HIGH discriminative power (mean separation: {mean_diff:.3f})")
        elif mean_diff > avg_std:
            print(f"  {feature}: MODERATE discriminative power (mean separation: {mean_diff:.3f})")
        else:
            print(f"  {feature}: LOW discriminative power (mean separation: {mean_diff:.3f})")
    
    return stats


def step4_bivariate_analysis(X, y, feature_names, visualizer):
    """
    Step 4: Perform unconditional bi-variate analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    y : np.ndarray
        Label vector (N,)
    feature_names : list
        Names of the features
    visualizer : Visualizer
        Visualizer instance
    """
    print_section_header("STEP 4: UNCONDITIONAL BI-VARIATE ANALYSIS")
    
    # Calculate correlation matrix
    correlation_matrix = StatisticalAnalyzer.calculate_correlation_matrix(X)
    
    print("\nCorrelation Matrix:")
    print("-" * 80)
    
    # Print correlation matrix with feature names
    header = "        " + "  ".join([f"{name[:8]:>8}" for name in feature_names])
    print(header)
    
    for i, feature in enumerate(feature_names):
        row = f"{feature[:8]:>8}"
        for j in range(len(feature_names)):
            row += f"  {correlation_matrix[i, j]:>8.3f}"
        print(row)
    
    # Identify strong correlations
    print("\nStrong Correlations (|ρ| > 0.7):")
    print("-" * 80)
    strong_corr_found = False
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(correlation_matrix[i, j]) > 0.7:
                strong_corr_found = True
                print(f"  {feature_names[i]} <-> {feature_names[j]}: ρ = {correlation_matrix[i, j]:.3f}")
    
    if not strong_corr_found:
        print("  No strong correlations found.")
    
    # Create visualizations
    print("\nGenerating scatter matrix...")
    visualizer.plot_scatter_matrix(X, y, feature_names)
    
    print("Generating correlation matrix heatmap...")
    visualizer.plot_correlation_matrix(correlation_matrix, feature_names)
    
    print("\n✓ Bi-variate analysis complete. Plots saved to outputs/figures/")
    
    return correlation_matrix


def step5_multivariate_analysis(X, y, visualizer):
    """
    Step 5: Perform unconditional multi-variate analysis (PCA).
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    y : np.ndarray
        Label vector (N,)
    visualizer : Visualizer
        Visualizer instance
    """
    print_section_header("STEP 5: UNCONDITIONAL MULTI-VARIATE ANALYSIS (PCA)")
    
    print("\nPerforming PCA (custom implementation)...")
    print("Note: Data will be standardized (mean=0, std=1) before PCA.")
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Print PCA summary
    pca.summary()
    
    # Create visualization
    print("\nGenerating PCA projection plot...")
    visualizer.plot_pca_projection(X_pca, y)
    
    print("\n✓ Multi-variate analysis complete. Plot saved to outputs/figures/")
    
    # Analysis of class separation
    print("\nClass Separation Analysis:")
    print("-" * 80)
    classes = np.unique(y)
    
    print("\nClass centroids in PC space:")
    for cls in classes:
        mask = y == cls
        centroid = np.mean(X_pca[mask], axis=0)
        print(f"  Class {cls}: PC1={centroid[0]:.3f}, PC2={centroid[1]:.3f}")
    
    # Calculate pairwise distances between class centroids
    print("\nPairwise distances between class centroids:")
    centroids = []
    for cls in classes:
        mask = y == cls
        centroid = np.mean(X_pca[mask], axis=0)
        centroids.append(centroid)
    
    for i, cls_i in enumerate(classes):
        for j, cls_j in enumerate(classes):
            if i < j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                print(f"  Class {cls_i} <-> Class {cls_j}: {dist:.3f}")
    
    return pca, X_pca


def main():
    """Main function to run all analysis steps."""
    print("\n" + "=" * 80)
    print("HOMEWORK 1: EXPLORATORY DATA ANALYSIS".center(80))
    print("=" * 80)
    
    # Initialize components
    data_loader = DataLoader(database_path="/database")
    visualizer = Visualizer(output_dir="homework1/outputs/figures")
    
    # Ensure output directory exists
    os.makedirs("homework1/outputs/figures", exist_ok=True)
    
    try:
        # Load data
        print("\nLoading data from /database directory...")
        
        # Try to find and load a dataset file
        if os.path.exists("/database"):
            files = [f for f in os.listdir("/database") 
                    if f.endswith(('.csv', '.txt', '.data'))]
            
            if not files:
                print("\nError: No dataset files found in /database directory.")
                print("Please place your dataset file (CSV, TXT, or DATA) in /database/")
                return
            
            # Use the first file found
            filename = files[0]
            print(f"Loading file: {filename}")
            
            data = data_loader.load_data(filename)
            print(f"Data loaded successfully! Shape: {data.shape}")
            
            # Extract features and labels (assuming last column is the label)
            X, y, feature_names = data_loader.extract_features_and_labels(data)
            
        else:
            print("\nError: /database directory not found.")
            print("Please create /database directory and place your dataset there.")
            return
        
        # Run all analysis steps
        step1_dataset_description(X, y, feature_names)
        
        step2_unconditional_analysis(X, feature_names, visualizer)
        
        step3_conditional_analysis(X, y, feature_names, visualizer)
        
        step4_bivariate_analysis(X, y, feature_names, visualizer)
        
        step5_multivariate_analysis(X, y, visualizer)
        
        # Final summary
        print_section_header("ANALYSIS COMPLETE")
        print("\nAll analysis steps have been completed successfully!")
        print(f"Results saved to: homework1/outputs/figures/")
        print("\nGenerated files:")
        print("  - unconditional_histograms.png")
        print("  - unconditional_boxplots.png")
        print("  - conditional_histograms.png")
        print("  - conditional_boxplots.png")
        print("  - scatter_matrix.png")
        print("  - correlation_matrix.png")
        print("  - pca_projection.png")
        print("\n" + "=" * 80)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. The /database directory exists")
        print("  2. Your dataset file is in /database/")
        print("  3. The file format is CSV, TXT, or DATA")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
