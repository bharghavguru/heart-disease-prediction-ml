
import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# Check zero values in numerical columns
columns = ["trestbps", "chol", "thalch", "oldpeak"]

for column in columns:

    zero_count = (df[column] == 0).sum()

    print("\nColumn:", column)
    print("Zero values:", zero_count)

# Check cholesterol values for both groups
print("\nCholesterol zero values by target:")
print(df[df["chol"] == 0]["target"].value_counts())

# Check gender distribution
print("\nGender distribution:")
print(df.groupby("sex")["target"].value_counts())
