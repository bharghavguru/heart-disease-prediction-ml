
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/raw/heart_disease_uci.csv")

# Get hospital names
hospitals = df["dataset"].unique()

print("Cholesterol zero values by hospital:")

for hospital in hospitals:

    hospital_data = df[df["dataset"] == hospital]

    zero_count = (hospital_data["chol"] == 0).sum()

    print(hospital, ":", zero_count)

# Check zero cholesterol values by target
print("\nZero cholesterol values by target:")

zero_cholesterol = df[df["chol"] == 0]

print(zero_cholesterol["num"].value_counts())

# Display some records with zero cholesterol
print("\nSample records with zero cholesterol:")

print(zero_cholesterol[["age", "sex", "dataset", "chol", "num"]].head(10))

# Check zero resting blood pressure
print("\nRecords with zero resting blood pressure:")

print(df[df["trestbps"] == 0][["age", "dataset", "trestbps", "num"]])
