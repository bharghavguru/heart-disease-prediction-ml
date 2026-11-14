
import pandas as pd

# Read the heart disease dataset
df = pd.read_csv("data/raw/heart_disease_uci.csv")

# Display the first five records
print("\nFirst 5 rows:")
print(df.head())

# Check dataset size
print("\nDataset size:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Check column names and data types
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
missing = df.isnull().sum()
print(missing)

# Check missing value percentages
print("\nMissing value percentages:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))

# Check duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# Display basic statistical information
print("\nDataset summary:")
print(df.describe(include="all"))

# Check original target distribution
print("\nOriginal heart disease distribution:")
print(df["num"].value_counts().sort_index())

# Convert target values into binary categories for analysis
binary_target = df["num"].apply(lambda value: 0 if value == 0 else 1)

print("\nBinary target distribution:")
print(binary_target.value_counts().rename({
    0: "No Heart Disease",
    1: "Heart Disease"
}))

# Display binary target percentages
print("\nBinary target percentages:")
print(
    binary_target.value_counts(normalize=True)
    .mul(100)
    .rename({
        0: "No Heart Disease",
        1: "Heart Disease"
    })
    .round(2)
)
