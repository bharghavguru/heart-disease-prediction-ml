# Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning classification algorithms to predict the presence of heart disease based on patient health-related attributes.

The project uses the UCI Heart Disease dataset and compares three machine learning algorithms to identify the best-performing model.

**Note:** This project is developed for educational purposes only. It is not intended for medical diagnosis or clinical decision-making.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset

- Dataset: UCI Heart Disease Dataset
- Original records: 920
- Original features: 16 columns
- Target: Heart disease presence or absence

The original target values were converted into binary classes:

- 0 – No Heart Disease
- 1 – Heart Disease

## Project Workflow

1. Dataset inspection and analysis
2. Data cleaning and handling missing values
3. Exploratory Data Analysis (EDA)
4. Feature preprocessing and encoding
5. Training machine learning models
6. Model evaluation and comparison
7. Saving the trained models
8. Predicting results using new input data

## Machine Learning Algorithms

Three classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Performance

The models were evaluated using an 80:20 stratified train-test split.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 78.80% | 78.90% | 84.31% | 81.52% |
| Decision Tree | 76.09% | 77.36% | 80.39% | 78.85% |
| Random Forest | 81.52% | 80.36% | 88.24% | 84.11% |

Random Forest achieved the best overall results among the three tested models.

## Project Structure

```text
AI-Heart-Disease-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│
├── inspect_dataset.py
├── analyze_missing.py
├── prepare_dataset.py
├── eda.py
├── preprocess.py
├── train_model.py
├── evaluate_model.py
├── predict.py
│
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd AI-Heart-Disease-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run preprocessing

```bash
python prepare_dataset.py
python preprocess.py
```

### 5. Train the models

```bash
python train_model.py
```

### 6. Evaluate the models

```bash
python evaluate_model.py
```

### 7. Run predictions

```bash
python predict.py
```

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Explore additional feature engineering techniques.
- Validate the model on independent datasets.
- Improve input validation in the prediction system.

## Disclaimer

This project is intended only for learning and demonstration. Its predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.