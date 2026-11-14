# Heart Disease Prediction Using Machine Learning

## Project Overview

This project aims to develop a Machine Learning model that predicts the presence of heart disease using patient health-related information.

The project uses the UCI Heart Disease dataset containing 920 patient records.

## Dataset Information

- Dataset: UCI Heart Disease Dataset
- Total records: 920
- Total features: 16
- Duplicate records: 0
- Target column: num

### Initial Dataset Findings

- 411 records belong to the no-heart-disease class.
- 509 records belong to the heart-disease class.
- The target variable will be converted into binary classification.
- The columns ca and thal contain more than 50% missing values.
- Missing values were identified for further preprocessing.

## Objectives

- Understand and explore the heart disease dataset.
- Clean and preprocess the data.
- Perform Exploratory Data Analysis (EDA).
- Build classification models using Machine Learning.
- Compare different models and evaluate their performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
AI-Heart-Disease-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
├── inspect_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Current Progress

- [x] Project folder setup
- [x] Python virtual environment setup
- [x] Required libraries installed
- [x] Dataset downloaded
- [x] Initial dataset inspection script

## Future Work

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Machine Learning model development
- Model evaluation and comparison
- Feature importance analysis

*Note: This project is for educational purposes and is not intended for medical diagnosis.*
