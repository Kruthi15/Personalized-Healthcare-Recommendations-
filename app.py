from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Recommendation mapping
recommendation_map = {
    0: "No Action",
    1: "Regular Checkup"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    recency = int(request.form['recency'])
    frequency = int(request.form['frequency'])
    monetary = int(request.form['monetary'])
    time = int(request.form['time'])
    
    # Create DataFrame
    patient_data = pd.DataFrame([[recency, frequency, monetary, time]],
                                columns=['Recency', 'Frequency', 'Monetary', 'Time'])
    
    # Scale input
    patient_scaled = scaler.transform(patient_data)
    
    # Predict
    prediction = model.predict(patient_scaled)[0]
    recommendation = recommendation_map.get(prediction, "Unknown")
    
    return render_template('result.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
