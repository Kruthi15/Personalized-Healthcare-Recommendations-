# Personalized-Healthcare-Recommendations-
# Personalized Healthcare Recommendation System 🩺

A machine learning-based web app that provides personalized healthcare recommendations using blood donation data. Built with Flask, XGBoost, and deployed locally with a clean, Bootstrap-based UI.

---

## 💡 Project Goal

To predict whether a patient needs a healthcare recommendation such as a regular checkup, based on personal health metrics like:

- Recency (days since last donation)
- Frequency (number of donations)
- Monetary (amount of blood donated)
- Time (months since first donation)

---

## 🔧 Tech Stack

- **Machine Learning**: XGBoost Classifier
- **Web App**: Flask
- **Frontend**: HTML, Bootstrap 5, CSS
- **Libraries**: pandas, scikit-learn, joblib, xgboost

---

## 🧠 ML Pipeline

1. Data Cleaning & Preprocessing
2. Feature Scaling (StandardScaler)
3. Model Training & Tuning with GridSearchCV
4. Model Evaluation (Accuracy, Confusion Matrix, ROC-AUC)
5. Recommendation Logic Mapping

---

## 💻 How to Run

```bash
pip install -r requirements.txt
python app.py

Access the app at http://127.0.0.1:5000/

