
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the prepared dataset
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# Set the graph style
sns.set_style("whitegrid")

# 1. Heart disease distribution
sns.countplot(data=df, x="target")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()


# 2. Age distribution
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# 3. Heart disease based on gender
sns.countplot(data=df, x="sex", hue="target")

plt.title("Heart Disease Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.show()


# 4. Cholesterol distribution
sns.boxplot(data=df, x="target", y="chol")

plt.title("Cholesterol Levels and Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol")
plt.show()


# 5. Maximum heart rate distribution
sns.boxplot(data=df, x="target", y="thalch")

plt.title("Maximum Heart Rate and Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Maximum Heart Rate")
plt.show()


# 6. Correlation between numerical features
numerical_data = df.select_dtypes(include="number")

correlation = numerical_data.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()
