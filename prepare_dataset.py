
import pandas as pd
import numpy as np
import os

# Load the original dataset
df = pd.read_csv("data/raw/heart_disease_uci.csv")

# Replace invalid zero values with missing values
df.loc[df["chol"] == 0, "chol"] = np.nan
df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

# Convert target into binary classification
df["target"] = df["num"].apply(lambda value: 1 if value > 0 else 0)

# Remove unnecessary columns
df = df.drop(columns=["id", "dataset", "ca", "thal", "num"])

# Display updated dataset
print("Updated Dataset:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display zero values in cholesterol and blood pressure
print("\nZero values after cleaning:")
print("Cholesterol:", (df["chol"] == 0).sum())
print("Resting Blood Pressure:", (df["trestbps"] == 0).sum())

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/heart_disease_cleaned.csv", index=False)

print("\nDataset saved successfully!")
