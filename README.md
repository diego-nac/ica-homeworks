# ica-homeworks

ica-homeworks contains the implementation of the homeworks for the discipline of Applied Computational Intelligence. The project applies computational intelligence techniques to solve the proposed exercises, structured and managed under the Scrum methodology to ensure iterative development and organized task execution.

## Project Structure

```
ica-homeworks/
├── database/                  # Database and data loading module
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   └── README.md
├── homework1/                 # Homework 1: Data Preprocessing
│   ├── preprocessing.ipynb   # Jupyter notebook for preprocessing
│   └── README.md
├── homework2/                 # Homework 2: Data Processing
│   ├── processing.ipynb      # Jupyter notebook for model training
│   └── README.md
├── homework3/                 # Homework 3: Data Analysis
│   ├── analysis.ipynb        # Jupyter notebook for analysis
│   └── README.md
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

## Homeworks Overview

### Homework 1: Data Preprocessing
Focus on data cleaning, feature engineering, and preparing data for machine learning models.
- Data loading and exploration
- Handling missing values and duplicates
- Feature scaling and normalization
- Encoding categorical variables

### Homework 2: Data Processing
Focus on machine learning model training and evaluation.
- Feature engineering and selection
- Model training with multiple algorithms
- Cross-validation and hyperparameter tuning
- Model evaluation and comparison

### Homework 3: Data Analysis
Focus on comprehensive analysis and interpretation of results.
- Exploratory Data Analysis (EDA)
- Statistical analysis and hypothesis testing
- Model performance analysis
- Feature importance and visualization

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/diego-nac/ica-homeworks.git
cd ica-homeworks
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

Navigate to each homework directory and open the respective Jupyter notebook:
- `homework1/preprocessing.ipynb` - Start with data preprocessing
- `homework2/processing.ipynb` - Continue with model training
- `homework3/analysis.ipynb` - Finish with data analysis

Each notebook is self-contained and includes detailed comments and explanations.

## Database Module

The `database` module provides utilities for data management:

```python
from database.data_loader import DataLoader

# Initialize the loader
loader = DataLoader(data_path='./data')

# Load data
df = loader.load_csv('dataset.csv')

# Save processed data
loader.save_csv(df, 'output.csv')
```

## Contributing

This project follows the Scrum methodology for iterative development. Contributions should maintain code quality and documentation standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
