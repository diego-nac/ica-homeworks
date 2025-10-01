# ica-homeworks
ica-homeworks contains the implementation of the homeworks for the discipline of Applied Computational Intelligence. The project applies computational intelligence techniques to solve the proposed exercises, structured and managed under the Scrum methodology to ensure iterative development and organized task execution.

## Homeworks

### Homework 1: Exploratory Data Analysis

Located in `homework1/`, this homework implements a comprehensive exploratory data analysis framework for classification datasets.

**Features:**
- Dataset description (N observations, D predictors, L classes)
- Unconditional mono-variate analysis (histograms, box-plots, statistics)
- Class-conditional mono-variate analysis
- Unconditional bi-variate analysis (scatter plots, correlation matrix)
- Custom PCA implementation (without using pre-made libraries)

**Usage:**
```bash
cd homework1
pip install -r requirements.txt

# Run with your dataset in /database directory
python main.py

# Or run the example with a sample dataset
python example.py
```

See [homework1/README.md](homework1/README.md) for detailed documentation.

## Project Structure

```
ica-homeworks/
├── homework1/          # Homework 1: Exploratory Data Analysis
│   ├── src/           # Source modules
│   ├── outputs/       # Generated outputs and figures
│   ├── main.py        # Main analysis script
│   ├── example.py     # Example with sample dataset
│   └── README.md      # Detailed documentation
├── README.md          # This file
└── LICENSE            # MIT License
```

## Requirements

- Python 3.7 or higher
- Dependencies are specified in each homework's `requirements.txt` file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
