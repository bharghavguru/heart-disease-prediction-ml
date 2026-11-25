
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Get the original feature columns used during training
train_data = pd.read_csv("data/processed/train_data.csv")
feature_columns = train_data.drop("target", axis=1).columns

print("===================================")
print(" HEART DISEASE PREDICTION SYSTEM")
print("===================================")

# Get patient details
age = int(input("Enter age: "))
sex = input("Enter sex (Male/Female): ")
sex = sex.strip().title()

cp = input(
    "Enter chest pain type "
    "(asymptomatic/atypical angina/non-anginal/typical angina): "
)
cp = cp.strip().lower()

trestbps = float(input("Enter resting blood pressure: "))
chol = float(input("Enter cholesterol level: "))

fbs_input = input("Fasting blood sugar > 120? (True/False): ")
if fbs_input.lower() == "true":
    fbs = True
else:
    fbs = False

restecg = input(
    "Enter resting ECG "
    "(lv hypertrophy/normal/st-t abnormality): "
)
restecg = restecg.strip().lower()

thalch = float(input("Enter maximum heart rate: "))

exang_input = input("Exercise-induced angina? (True/False): ")
if exang_input.lower() == "true":
    exang = True
else:
    exang = False

oldpeak = float(input("Enter ST depression (oldpeak): "))

slope = input(
    "Enter slope (downsloping/flat/upsloping): "
)
slope = slope.strip().lower()

# Create a dictionary with patient details
patient_data = {
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalch": [thalch],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope]
}

# Convert patient details into a DataFrame
patient_df = pd.DataFrame(patient_data)

# Convert categorical values into numerical columns
patient_df = pd.get_dummies(patient_df)

# Match the columns used during model training
patient_df = patient_df.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict using the trained model
prediction = model.predict(patient_df)

print("\n===================================")
print(" PREDICTION RESULT")
print("===================================")

if prediction[0] == 1:
    print("Prediction: Heart Disease")
else:
    print("Prediction: No Heart Disease")

print("\nNote: This is an educational ML project.")
print("This prediction is not a medical diagnosis.")
