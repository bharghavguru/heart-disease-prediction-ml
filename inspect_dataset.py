
import pandas as pd

# Read the heart disease dataset
df = pd.read_csv("data/raw/heart_disease_uci.csv")

# Take a look at the data
print("\nFirst 5 rows:")
print(df.head())

# Check how many rows and columns are present
print("\nDataset size:")
print(df.shape)

# Check the column names and data types
print("\nDataset information:")
df.info()

# Find missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# Check whether any records are repeated
print("\nDuplicate records:")
print(df.duplicated().sum())

# Get a basic summary of the dataset
print("\nDataset summary:")
print(df.describe(include="all"))

# Check the number of records in each target class
print("\nHeart disease distribution:")
print(df["num"].value_counts(dropna=False))
