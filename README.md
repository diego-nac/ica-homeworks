# ica-homeworks

ica-homeworks contains the implementation of the homeworks for the discipline of Applied Computational Intelligence. The project applies computational intelligence techniques to solve the proposed exercises, structured and managed under the Scrum methodology to ensure iterative development and organized task execution.

## Project Structure

```
ica-homeworks/
├── database/                  # Database and data loading module
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   └── README.md
├── homework1/                 # Homework 1: Data Analysis
│   ├── preprocessing.ipynb   # Jupyter notebook for data analysis
│   └── README.md
├── homework2/                 # Homework 2: Methods for Regression
│   ├── processing.ipynb      # Jupyter notebook for regression methods
│   └── README.md
├── homework3/                 # Homework 3: Methods for Classification
│   ├── analysis.ipynb        # Jupyter notebook for classification methods
│   └── README.md
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

## Dataset

This project uses the **Air Quality Index** dataset, which contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

## Homeworks Overview

All homeworks follow the IEEE (Institute of Electrical and Electronics Engineers) Manuscript Templates for Conference Proceedings format. Papers must not exceed 6 pages and include the following sections:
- **Abstract**: Main objective and overview of the work
- **Introduction**: Context, background, and literature review
- **Methods**: Explanation of methods/algorithms used
- **Results**: Discussion of results

### Homework 1: Data Analysis
Focus on exploratory data analysis and understanding the Air Quality dataset.
- Data loading and exploration
- Statistical analysis and visualization
- Data quality assessment
- Feature understanding and correlation analysis
- Initial insights and patterns

### Homework 2: Methods for Regression
Focus on applying regression methods to predict continuous variables in the Air Quality dataset.
- Feature engineering and selection
- Regression model implementation (Linear Regression, Ridge, Lasso, Random Forest, etc.)
- Model training and hyperparameter tuning
- Cross-validation techniques
- Model evaluation and comparison

### Homework 3: Methods for Classification
Focus on applying classification methods to categorize air quality levels.
- Feature preprocessing for classification
- Classification model implementation (Logistic Regression, SVM, Decision Trees, Random Forest, etc.)
- Model training and optimization
- Performance metrics (accuracy, precision, recall, F1-score)
- Model comparison and selection

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
- `homework1/preprocessing.ipynb` - Start with data analysis
- `homework2/processing.ipynb` - Continue with regression methods
- `homework3/analysis.ipynb` - Finish with classification methods

Each notebook is self-contained and includes detailed comments and explanations following the IEEE paper format structure.

## Database Module

The `database` module provides utilities for data management:

```python
from database.data_loader import DataLoader

# Initialize the loader
loader = DataLoader(data_path='./data')

# Load Air Quality data
df = loader.load_csv('AirQualityUCI.csv')

# Save processed data
loader.save_csv(df, 'output.csv')
```

## Paper Format Guidelines

Each homework should follow the IEEE conference paper format:
- Maximum 6 pages
- Include: Abstract, Introduction, Methods, Results sections
- Provide references and citations
- Work can be done individually or with max 4 co-authors
- Must include a clear statement of each contributor's work

## Contributing

This project follows the Scrum methodology for iterative development. Contributions should maintain code quality and documentation standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
