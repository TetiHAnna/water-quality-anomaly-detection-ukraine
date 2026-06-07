# Machine Learning-Based Anomaly Detection in Surface Water Quality Data

This repository contains the source code and processed dataset used in the manuscript:

"Machine Learning-Based Anomaly Detection in Surface Water Quality Data Using Ensemble Models with Residual Analysis"

## Authors

Tetiana Nosenko  
Iryna Mashkina

## Dataset

The original monitoring data were obtained from the State Water Resources Agency of Ukraine and processed for machine-learning analysis.

## Software Environment

- Python 3.14.2
- scikit-learn 1.9.0
- XGBoost 3.2.0
- pandas
- numpy
- matplotlib

## Repository Structure

- `01_model_validation.py` — model validation and cross-validation
- `cv_results_table2.csv` — cross-validation results
- `combined_water_quality.csv` — processed dataset

## Reproducibility

## Reproducibility

Run:

```bash
python 01_model_validation.py
```

to reproduce the cross-validation results reported in the manuscript.

## Data Source

State Water Resources Agency of Ukraine:

https://data.gov.ua/dataset/surface-water-monitoring
