"""
Example script demonstrating homework1 analysis with a sample dataset.

This script creates a sample dataset and runs the analysis.
For actual use, place your dataset in /database/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from statistical_analysis import StatisticalAnalyzer
from visualization import Visualizer
from pca import PCA


def create_sample_dataset(n_samples=150, save_path="database"):
    """
    Create a sample dataset similar to Iris dataset for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    save_path : str
        Directory to save the dataset
    """
    np.random.seed(42)
    
    # Create three classes with different distributions
    n_per_class = n_samples // 3
    
    # Class 0: Small flowers
    class0_f1 = np.random.normal(5.0, 0.3, n_per_class)
    class0_f2 = np.random.normal(3.4, 0.2, n_per_class)
    class0_f3 = np.random.normal(1.5, 0.2, n_per_class)
    class0_f4 = np.random.normal(0.2, 0.1, n_per_class)
    
    # Class 1: Medium flowers
    class1_f1 = np.random.normal(6.0, 0.4, n_per_class)
    class1_f2 = np.random.normal(2.8, 0.3, n_per_class)
    class1_f3 = np.random.normal(4.5, 0.4, n_per_class)
    class1_f4 = np.random.normal(1.4, 0.2, n_per_class)
    
    # Class 2: Large flowers
    class2_f1 = np.random.normal(6.5, 0.4, n_per_class)
    class2_f2 = np.random.normal(3.0, 0.3, n_per_class)
    class2_f3 = np.random.normal(5.5, 0.5, n_per_class)
    class2_f4 = np.random.normal(2.0, 0.3, n_per_class)
    
    # Combine into dataset
    feature1 = np.concatenate([class0_f1, class1_f1, class2_f1])
    feature2 = np.concatenate([class0_f2, class1_f2, class2_f2])
    feature3 = np.concatenate([class0_f3, class1_f3, class2_f3])
    feature4 = np.concatenate([class0_f4, class1_f4, class2_f4])
    labels = np.concatenate([np.zeros(n_per_class), 
                            np.ones(n_per_class), 
                            np.full(n_per_class, 2)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'sepal_length': feature1,
        'sepal_width': feature2,
        'petal_length': feature3,
        'petal_width': feature4,
        'class': labels.astype(int)
    })
    
    # Save dataset
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, 'sample_dataset.csv')
    df.to_csv(filepath, index=False)
    print(f"Sample dataset created: {filepath}")
    
    return df


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def main():
    """Run example analysis with sample dataset."""
    print("\n" + "=" * 80)
    print("HOMEWORK 1 EXAMPLE: EXPLORATORY DATA ANALYSIS".center(80))
    print("=" * 80)
    
    # Create sample dataset in local database directory
    print("\nCreating sample dataset...")
    create_sample_dataset(n_samples=150, save_path="homework1/database")
    
    # Initialize components
    data_loader = DataLoader(database_path="homework1/database")
    visualizer = Visualizer(output_dir="homework1/outputs/figures")
    
    # Ensure output directory exists
    os.makedirs("homework1/outputs/figures", exist_ok=True)
    
    # Load data
    print("\nLoading sample dataset...")
    data = data_loader.load_data('sample_dataset.csv')
    print(f"Data loaded successfully! Shape: {data.shape}")
    
    # Extract features and labels
    X, y, feature_names = data_loader.extract_features_and_labels(data)
    
    # Step 1: Dataset Description
    print_section_header("STEP 1: DATASET DESCRIPTION")
    info = data_loader.get_dataset_info(X, y)
    print(f"\nNumber of observations (N): {info['N']}")
    print(f"Number of predictor variables (D): {info['D']}")
    print(f"Number of classes (L): {info['L']}")
    print(f"\nFeature names: {feature_names}")
    print(f"\nClass distribution:")
    for cls, count in info['class_counts'].items():
        percentage = (count / info['N']) * 100
        print(f"  Class {cls}: {count} observations ({percentage:.2f}%)")
    
    # Step 2: Unconditional Analysis
    print_section_header("STEP 2: UNCONDITIONAL MONO-VARIATE ANALYSIS")
    stats = StatisticalAnalyzer.unconditional_monovariate_analysis(X, feature_names)
    print(StatisticalAnalyzer.format_statistics_table(stats))
    
    print("\nGenerating unconditional visualizations...")
    visualizer.plot_unconditional_histograms(X, feature_names)
    visualizer.plot_unconditional_boxplots(X, feature_names)
    print("✓ Unconditional analysis complete.")
    
    # Step 3: Conditional Analysis
    print_section_header("STEP 3: CLASS-CONDITIONAL MONO-VARIATE ANALYSIS")
    cond_stats = StatisticalAnalyzer.conditional_monovariate_analysis(X, y, feature_names)
    print(StatisticalAnalyzer.format_conditional_statistics_table(cond_stats))
    
    print("\nGenerating class-conditional visualizations...")
    visualizer.plot_conditional_histograms(X, y, feature_names)
    visualizer.plot_conditional_boxplots(X, y, feature_names)
    print("✓ Class-conditional analysis complete.")
    
    # Step 4: Bi-variate Analysis
    print_section_header("STEP 4: UNCONDITIONAL BI-VARIATE ANALYSIS")
    corr_matrix = StatisticalAnalyzer.calculate_correlation_matrix(X)
    
    print("\nCorrelation Matrix:")
    print("-" * 80)
    header = "        " + "  ".join([f"{name[:8]:>8}" for name in feature_names])
    print(header)
    for i, feature in enumerate(feature_names):
        row = f"{feature[:8]:>8}"
        for j in range(len(feature_names)):
            row += f"  {corr_matrix[i, j]:>8.3f}"
        print(row)
    
    print("\nGenerating bi-variate visualizations...")
    visualizer.plot_scatter_matrix(X, y, feature_names)
    visualizer.plot_correlation_matrix(corr_matrix, feature_names)
    print("✓ Bi-variate analysis complete.")
    
    # Step 5: Multi-variate Analysis (PCA)
    print_section_header("STEP 5: UNCONDITIONAL MULTI-VARIATE ANALYSIS (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca.summary()
    
    print("\nGenerating PCA projection plot...")
    visualizer.plot_pca_projection(X_pca, y)
    print("✓ Multi-variate analysis complete.")
    
    # Final Summary
    print_section_header("ANALYSIS COMPLETE")
    print("\nAll analysis steps completed successfully!")
    print(f"\nResults saved to: homework1/outputs/figures/")
    print("\nGenerated files:")
    print("  - unconditional_histograms.png")
    print("  - unconditional_boxplots.png")
    print("  - conditional_histograms.png")
    print("  - conditional_boxplots.png")
    print("  - scatter_matrix.png")
    print("  - correlation_matrix.png")
    print("  - pca_projection.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
