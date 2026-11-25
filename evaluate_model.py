
import pandas as pd
import joblib
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load testing data
test_data = pd.read_csv("data/processed/test_data.csv")

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# List of trained models
model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

model_files = [
    "models/logistic_regression.pkl",
    "models/decision_tree.pkl",
    "models/random_forest.pkl"
]

# Evaluate each model
for i in range(len(model_names)):

    print("\n===================================")
    print(model_names[i])
    print("===================================")

    # Load the trained model
    model = joblib.load(model_files[i])

    # Predict results
    predictions = model.predict(X_test)

    # Display confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Heart Disease", "Heart Disease"]
    ))
