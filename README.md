# CancerRegression

## Overview

This repository contains a regression analysis project for predicting cancer death rates using demographic and healthcare data.

The project merges two CSV datasets on the `geography` field and trains several regression models to compare performance.

## Data source

The original dataset is from Kaggle:

- https://www.kaggle.com/datasets/varunraskar/cancer-regression

The repository includes the local dataset files:

- `cancer-regression.csv`
- `avg-household-size.csv`

## Files

- `benchmark_models.py` - trains and evaluates a set of regression models on the merged dataset.
- `verify_models.py` - runs a selected model suite and prints model metrics for comparison.
- `cancerregression.ipynb` - Jupyter notebook for exploratory analysis and model experimentation.
- `cancerregression.html` - exported HTML view of the notebook.
- `fix_notebook.py` / `update_notebook.py` - utility scripts for notebook maintenance.
- `README.md` - project documentation.

## Requirements

This project uses Python and the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost

Install dependencies with pip:

```powershell
pip install pandas numpy scikit-learn xgboost
```

## Usage

Run the benchmark script:

```powershell
cd /d D:\projects\cancerregression
python benchmark_models.py
```

Run the verification script:

```powershell
cd /d D:\projects\cancerregression
python verify_models.py
```

## Notes

- The repository currently uses a local `master` branch in the working copy.
- If you want to update the remote upstream or origin, make sure to push the correct branch name.
- `benchmark_models.py` and `verify_models.py` both require the CSV data files to be present in the same directory.

