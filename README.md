# 📈 Company Success Prediction Using Machine Learning

A beginner-friendly Machine Learning project that predicts a company's **Success Rate**, **Failure Rate**, and **Survival Rate** based on company details such as company name, number of employees, and investment amount.

This project demonstrates the complete Machine Learning workflow, including data preprocessing, model training, evaluation, visualization, and prediction using Python.

---

## 🚀 Features

- Predicts:
  - ✅ Success Rate (%)
  - ❌ Failure Rate (%)
  - 📊 Survival Rate (%)
- Uses Random Forest Regressor
- Label Encoding for company names
- Automatic data preprocessing
- Clean dataset (no missing values or duplicates)
- Model evaluation using:
  - R² Score
  - Mean Absolute Error (MAE)
- Feature Importance Visualization
- Saves the trained model using Joblib
- Beginner-friendly and well-commented code

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 📂 Project Structure

```
Company-Success-Prediction/
│
├── company.csv          # Dataset
├── main.py              # Main program
├── model.pkl            # Saved Machine Learning model
├── encoder.pkl          # Label Encoder
├── README.md            # Project Documentation
└── requirements.txt     # Required Python libraries
```

---

## 📊 Dataset

The dataset contains **200 realistic company records** with the following columns:

- Company
- Employees
- Investment
- Success_Rate
- Failure_Rate
- Survival_Rate

The dataset is clean and contains:
- No missing values
- No duplicate records
- Realistic numerical values

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Display Dataset
3. Verify Data Quality
4. Encode Company Names
5. Split Training & Testing Data
6. Train Random Forest Regressor
7. Evaluate Model
8. Display Feature Importance
9. Save Trained Model
10. Predict Company Performance

---

## 📈 Model Evaluation

The model is evaluated using:

- R² Score
- Mean Absolute Error (MAE)

These metrics help measure the accuracy of the predictions.

---
## 💻 Example Input

```
Company Name : TechNova
Employees    : 250
Investment   : 5000000
```

### Example Output

```
Predicted Success Rate  : 84.27%
Predicted Failure Rate  : 10.43%
Predicted Survival Rate : 89.57%
```

---

## 📌 Learning Outcomes

This project helped me understand:

- Data preprocessing
- Label Encoding
- Regression using Random Forest
- Model evaluation
- Feature Importance
- Saving and loading models
- Building an end-to-end Machine Learning project

---

## 🔮 Future Improvements

- Build a web interface using Flask or Streamlit
- Support CSV file prediction for multiple companies
- Compare multiple Machine Learning algorithms
- Deploy the model online
- Add interactive charts and dashboards

---

## 👨‍💻 Author
Salman Afridi
If you found this project helpful, feel free to ⭐ star this repository

**Shifa Fathima**

If you found this project helpful, feel free to ⭐ star this repository!
