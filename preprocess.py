
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# Separate input features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Identify numerical and categorical columns
numerical_columns = ["age", "trestbps", "chol", "thalch", "oldpeak"]

categorical_columns = [
    "sex", "cp", "fbs", "restecg", "exang", "slope"
]

# Fill missing numerical values using median
for column in numerical_columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)

# Fill missing categorical values using mode
for column in categorical_columns:

    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(mode_value)
    X_test[column] = X_test[column].fillna(mode_value)

# Convert categorical values into numerical values
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Make sure both datasets have the same columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Check missing values
print("\nMissing values in training data:")
print(X_train.isnull().sum().sum())

print("\nMissing values in testing data:")
print(X_test.isnull().sum().sum())

# Display final dataset shapes
print("\nFinal Training Shape:", X_train.shape)
print("Final Testing Shape:", X_test.shape)

# Save the prepared datasets
os.makedirs("data/processed", exist_ok=True)

train_data = X_train.copy()
train_data["target"] = y_train

test_data = X_test.copy()
test_data["target"] = y_test

train_data.to_csv("data/processed/train_data.csv", index=False)
test_data.to_csv("data/processed/test_data.csv", index=False)

print("\nPreprocessing completed successfully!")
