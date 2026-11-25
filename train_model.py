import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Load the training and testing datasets
train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

# Separate input features and target
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Make sure training and testing have the same feature columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# Create the three machine learning models

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

decision_tree_model = DecisionTreeClassifier(random_state=42)

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Store the models for training
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


# Create folders to store the models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_results = []


# Train and evaluate each model
for model_name, model in models.items():

    print("\nTraining:", model_name)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions using testing data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Display the results
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    # Store the results
    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Save the trained model
    model_file_name = model_name.lower().replace(" ", "_")
    joblib.dump(model, "models/" + model_file_name + ".pkl")


# Create a comparison table
results_df = pd.DataFrame(model_results)

print("\nModel Comparison:")
print(results_df.round(4))

# Save the comparison results
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nAll models trained and saved successfully!")
print("Model files are available in the models folder.")
print("Comparison results are available in the results folder.")