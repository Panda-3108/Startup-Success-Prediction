import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("company.csv")

print("===================================")
print("   DATASET LOADED SUCCESSFULLY")
print("===================================\n")

print(df.head())

print("\n========== DATASET INFORMATION ==========\n")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())

# ==========================
# LABEL ENCODING
# ==========================

encoder = LabelEncoder()
df["Company"] = encoder.fit_transform(df["Company"])

print("\n========== ENCODED DATA ==========\n")
print(df.head())

# ==========================
# FEATURES & TARGETS
# ==========================

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

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# ==========================
# MODEL
# ==========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

# ==========================
# EVALUATION
# ==========================

predictions = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========\n")

print("R2 Score :", round(r2_score(Y_test, predictions), 3))
print("MAE      :", round(mean_absolute_error(Y_test, predictions), 3))

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = model.feature_importances_

plt.figure(figsize=(7,4))
plt.bar(X.columns, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(model, "company_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\nModel Saved Successfully!")

# ==========================
# USER INPUT
# ==========================

company_name = input("\nEnter Company Name : ")

employees = int(input("Enter Number of Employees : "))

investment = float(input("Enter Investment Amount : "))

company = encoder.transform([company_name])[0]

result = model.predict([[company, employees, investment]])

success = result[0][0]
failure = result[0][1]
survival = result[0][2]

# ==========================
# COMPANY REPORT
# ==========================

print("\n")
print("=" * 60)
print("             COMPANY PREDICTION REPORT")
print("=" * 60)

print(f"Company Name        : {company_name}")
print(f"Employees           : {employees}")
print(f"Investment Amount   : ₹{investment:,.2f}")

print("-" * 60)

print(f"Success Rate        : {success:.2f}%")
print(f"Failure Rate        : {failure:.2f}%")
print(f"Survival Rate       : {survival:.2f}%")

print("-" * 60)

# ==========================
# BUSINESS STATUS
# ==========================

if success >= 80 and survival >= 80:
    business_status = "Highly Stable Company ✅"

elif success >= 60 and survival >= 60:
    business_status = "Growing Company 🟢"

elif success >= 40:
    business_status = "Average Performance 🟡"

else:
    business_status = "Business Needs Improvement 🔴"

print(f"Business Status     : {business_status}")

# ==========================
# INVESTMENT RISK
# ==========================

if failure < 20:
    risk = "LOW 🟢"

elif failure < 40:
    risk = "MEDIUM 🟡"

else:
    risk = "HIGH 🔴"

print(f"Investment Risk     : {risk}")

print("=" * 60)
# ==========================
# RECOMMENDATION SYSTEM
# ==========================

print("\n")
print("=" * 60)
print("              BUSINESS RECOMMENDATIONS")
print("=" * 60)

# LOW RISK
if failure < 20:

    print("\nReason for Current Performance:")
    print("• Company has a very good success probability.")
    print("• Investment is sufficient for the business size.")
    print("• Employee strength supports business growth.")

    print("\nRecommended Actions:")
    print("✔ Continue investing in innovation.")
    print("✔ Expand into new markets.")
    print("✔ Improve employee skill development.")
  
    print("\nOverall Risk Level : LOW 🟢")
    print("Conclusion         : Company has excellent growth potential.")

# MEDIUM RISK
elif failure < 40:

    print("\nReason for Current Performance:")
    print("• Business is stable but has moderate risks.")
    print("• Investment can be improved.")
    print("• Employee productivity can be enhanced.")

    print("\nRecommended Actions:")
    print("✔ Increase investment gradually.")
    print("✔ Hire experienced professionals.")
    print("✔ Improve marketing strategies.")

    print("\nOverall Risk Level : MEDIUM 🟡")
    print("Conclusion         : Company has good potential with improvements.")

# HIGH RISK
else:

    print("\nReason for High Failure Rate:")
    print("• Predicted success rate is comparatively low.")
    print("• Investment appears insufficient for company growth.")
    print("• Employee strength may not be adequate for business needs.")
    print("\nRecommended Actions:")
    print("✔ Increase investment to strengthen operations.")
    print("✔ Recruit skilled employees.")
    print("✔ Improve product/service quality.")
    
    print("\nOverall Risk Level : HIGH 🔴")
    print("Conclusion         : Immediate business improvements are recommended.")

print("=" * 60)

# ==========================
# FINAL SUMMARY
# ==========================

print("\n")
print("=" * 60)
print("                FINAL SUMMARY")
print("=" * 60)

if success >= 80:
    print("🎉 Excellent! The company has a strong probability of success.")

elif success >= 60:
    print("👍 The company is performing well with scope for further growth.")

elif success >= 40:
    print("⚠ The company has average performance and needs improvements.")

else:
    print("❌ The company has a high chance of failure if no improvements are made.")

print("=" * 60)
print("      Thank You for Using Company Success Predictor")
print("=" * 60)

