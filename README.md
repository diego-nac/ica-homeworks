# ica-homeworks

Homeworks for the Applied Computational Intelligence course using the Air Quality UCI dataset. Each homework has a notebook and an IEEE-formatted report that documents the work.

## Structure (main files)

```
database/
	air_quality_uci.csv        # Raw dataset

homework1/
	code.ipynb                 # Exploratory data analysis and PCA
	doc/main.pdf               # IEEE report (built)

homework2/
	code.ipynb                 # Regression experiments
	doc/main.pdf               # IEEE report (built)

homework3/
	analysis.ipynb             # Classification experiments
	doc/main.pdf               # IEEE report (built if compiled)

requirements.txt
LICENSE
README.md
```

## Dataset

Air Quality UCI: 9,358 hourly samples from 5 metal-oxide sensors measuring pollutants. Stored in `database/air_quality_uci.csv` (original source: UCI Machine Learning Repository).

## Homework briefs

- `homework1/code.ipynb`: Exploratory analysis, distributions, correlations, and PCA to understand the dataset and sensor behavior. Figures are saved under `homework1/figures/`; the report lives in `homework1/doc/`.
- `homework2/code.ipynb`: Regression on pollutant targets (e.g., CO). Includes preprocessing, model fitting (linear and regularized baselines, plus other regressors), cross-validation, and metric comparison. Results and discussion are in `homework2/doc/`.
- `homework3/analysis.ipynb`: Classification of air-quality levels. Covers preprocessing, model training (logistic/SVM/tree ensembles, etc.), and evaluation with accuracy/F1/ROC-style metrics. Report and figures are in `homework3/doc/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## How to run

```bash
jupyter notebook
```

Open the notebook of interest (`homework1/code.ipynb`, `homework2/code.ipynb`, or `homework3/analysis.ipynb`) and run the cells. Each notebook is self-contained and expects the dataset at `database/air_quality_uci.csv`.

## License

MIT License. See `LICENSE`.
