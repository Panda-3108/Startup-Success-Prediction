import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
df = pd.read_csv("company.csv")

print("Dataset Loaded Successfully\n")

print(df.head())
print("\nDataset Information\n")

print(df.info())

print("\nStatistical Summary\n")

print(df.describe())
print("\nMissing Values\n")

print(df.isnull().sum())
print("\nDuplicate Rows")

print(df.duplicated().sum())
encoder = LabelEncoder()

df["Company"] = encoder.fit_transform(df["Company"])

print("\nEncoded Dataset\n")

print(df.head())
X = df[[
    "Company",
    "Employees",
    "Investment"
]]

Y = df[[
    "Success_Rate",
    "Failure_Rate",
    "Survival_Rate"
]]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)

print("\nModel Training Completed")

predictions = model.predict(X_test)
print("\nModel Evaluation")

print("R2 Score:", r2_score(Y_test, predictions))

print("MAE:", mean_absolute_error(Y_test, predictions))
importance = model.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(X.columns, importance)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()
joblib.dump(model, "company_model.pkl")

print("\nModel Saved Successfully")
company = input("\nEnter Company Name: ")

employees = int(input("Enter Number of Employees: "))

investment = float(input("Enter Investment Amount: "))
company = encoder.transform([company])[0]
result = model.predict([[company, employees, investment]])
print("\nPrediction Result")
print("-" * 30)

print(f"Success Rate : {result[0][0]:.2f}%")
print(f"Failure Rate : {result[0][1]:.2f}%")
print(f"Survival Rate: {result[0][2]:.2f}%")