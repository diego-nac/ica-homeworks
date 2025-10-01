# Homework 1: Exploratory Data Analysis

This homework implements a comprehensive exploratory data analysis framework for classification datasets. The analysis follows five main steps to understand the dataset structure, feature distributions, relationships, and class separability.

## Project Structure

```
homework1/
├── main.py                 # Main analysis script
├── src/                    # Source modules
│   ├── data_loader.py      # Data loading utilities
│   ├── statistics.py       # Statistical analysis functions
│   ├── visualization.py    # Visualization functions
│   └── pca.py             # Custom PCA implementation
├── outputs/                # Output directory
│   └── figures/           # Generated plots and figures
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

The analysis requires the following Python packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Location

**Important:** The dataset must be located in the `/database` directory at the root level.

Supported file formats:
- CSV (`.csv`)
- Text files (`.txt`)
- Data files (`.data`)

The script expects the dataset to have:
- Features in columns (except the last one)
- Class labels in the last column
- Header row with feature names (optional for `.data` files)

## Analysis Steps

The analysis performs the following five steps:

### Step 1: Dataset Description
- Identifies number of observations (N)
- Counts predictor variables (D)
- Determines number of classes (L)
- Reports class distribution

### Step 2: Unconditional Mono-variate Analysis
- Plots unconditional histograms for each feature
- Creates unconditional box-plots
- Calculates mean (μ_d), standard deviation (σ_d), and skewness (γ_d) for each feature
- Generates D histograms and D box-plots
- Tabulates all statistics

**Outputs:**
- `unconditional_histograms.png`
- `unconditional_boxplots.png`

### Step 3: Class-Conditional Mono-variate Analysis
- Plots class-conditional histograms for each feature
- Creates class-conditional box-plots
- Calculates class-conditional mean (μ_d|l), standard deviation (σ_d|l), and skewness (γ_d|l)
- Generates D×L histograms and D×L box-plots
- Tabulates all class-conditional statistics
- Analyzes discriminative power of each feature

**Outputs:**
- `conditional_histograms.png`
- `conditional_boxplots.png`

**Analysis:** The script identifies features with discriminative power by comparing mean separation relative to standard deviations across classes.

### Step 4: Unconditional Bi-variate Analysis
- Creates scatter plots for all pairs of predictors
- Uses colors/symbols to indicate class labels
- Calculates pair-wise correlation coefficients (ρ_di,dj)
- Displays correlation matrix as both table and heatmap
- Identifies strong correlations (|ρ| > 0.7)
- Investigates linear relationships and potential outliers

**Outputs:**
- `scatter_matrix.png`
- `correlation_matrix.png`

**Analysis:** The script quantifies linear dependence between predictors and highlights strong correlations that may indicate redundancy or multicollinearity.

### Step 5: Unconditional Multi-variate Analysis (PCA)
- Implements PCA from scratch without using pre-made libraries
- Performs necessary data preprocessing (standardization: mean=0, std=1)
- Calculates covariance matrix
- Computes eigenvalues and eigenvectors
- Retains first two principal components (PC1 and PC2)
- Projects observations onto PC space
- Plots scatter plot with class colors
- Reports explained variance ratio
- Analyzes class separation in reduced space

**Outputs:**
- `pca_projection.png`

**Analysis:** The script evaluates:
- Whether classes are well separated in PC space
- If class boundaries appear linear or non-linear
- Which classes show high overlap and are harder to separate
- Class centroids and pairwise distances in PC space

## Usage

### Basic Usage

```bash
# Navigate to the homework1 directory
cd homework1

# Run the analysis
python main.py
```

### Example Output

The script will:
1. Load the dataset from `/database`
2. Print detailed statistics and analysis results to console
3. Generate 7 visualization files in `outputs/figures/`
4. Provide insights about discriminative features and class separability

### Console Output

The analysis prints comprehensive results including:
- Dataset description with N, D, L values
- Unconditional statistics table (mean, std, skewness for each feature)
- Class-conditional statistics table (mean, std, skewness for each feature and class)
- Discriminative power analysis for each feature
- Correlation matrix
- Strong correlation pairs
- PCA summary with explained variance ratios
- Class separation analysis in PC space

## Implementation Details

### Custom PCA Implementation

The PCA module (`src/pca.py`) implements PCA from scratch:

1. **Standardization**: Centers data to mean=0 and scales to std=1
2. **Covariance Matrix**: Computes X^T X / (N-1)
3. **Eigendecomposition**: Calculates eigenvalues and eigenvectors
4. **Sorting**: Orders by eigenvalues (descending)
5. **Projection**: Projects data onto principal components

**Note:** No sklearn or other pre-made PCA functions are used.

### Statistical Analysis

All statistical calculations are performed using basic numpy and scipy functions:
- Mean: `np.mean()`
- Standard Deviation: `np.std(ddof=1)` for sample std
- Skewness: `scipy.stats.skew()`
- Correlation: `np.corrcoef()`

### Visualization

Visualizations use matplotlib and seaborn for professional-quality plots:
- Histograms show distribution shapes
- Box-plots reveal outliers and quartiles
- Scatter matrices show pairwise relationships
- Heatmaps visualize correlation structure
- PCA plots show class separation

## Customization

### Modifying the Analysis

You can customize the analysis by:

1. **Changing the database path**: Edit the `database_path` parameter in `main.py`
2. **Selecting different label column**: Modify `label_column` in `extract_features_and_labels()`
3. **Adjusting number of PCA components**: Change `n_components` in the PCA initialization
4. **Modifying visualization settings**: Edit parameters in `visualization.py`

### Adding New Features

The modular structure allows easy extension:
- Add new statistical measures in `statistics.py`
- Create new visualizations in `visualization.py`
- Implement additional analysis steps in `main.py`

## Troubleshooting

### Common Issues

1. **"No dataset files found in /database directory"**
   - Ensure `/database` directory exists at root level
   - Check that dataset file has correct extension (.csv, .txt, or .data)

2. **"File not found" errors**
   - Verify the database path is `/database` (absolute path)
   - Check file permissions

3. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure Python version >= 3.7

4. **Memory errors with large datasets**
   - Consider reducing number of bins in histograms
   - Sample the data before plotting scatter matrices

## Expected Outputs

After successful execution, you should have:

1. **Console Output**: Complete statistical analysis with tables
2. **Seven PNG files** in `outputs/figures/`:
   - Unconditional histograms and box-plots
   - Class-conditional histograms and box-plots
   - Scatter matrix showing all pairwise feature relationships
   - Correlation matrix heatmap
   - PCA projection plot

## References

This homework follows the requirements specified in the Applied Computational Intelligence course, implementing exploratory data analysis techniques for classification problems.

## License

This project is part of the ica-homeworks repository and follows the MIT License.
