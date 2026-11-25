
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/heart_disease_uci.csv")

# Display records from each hospital
print("Records in each hospital:")
print(df["dataset"].value_counts())

# Get the hospital names
hospitals = df["dataset"].unique()

# Check missing values in each hospital
for hospital in hospitals:

    hospital_data = df[df["dataset"] == hospital]

    print("\nHospital:", hospital)
    print("Total records:", len(hospital_data))

    print("\nMissing values:")

    for column in df.columns:

        missing = hospital_data[column].isnull().sum()

        if missing > 0:

            percentage = (missing / len(hospital_data)) * 100

            print(column, ":", missing,
                  "(", round(percentage, 2), "% )")
