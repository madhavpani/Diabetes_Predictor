from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Load the data
df = pd.read_csv('diabetes_probability.csv')

# Load Trained Model
model = joblib.load('Diabetes_Probability_Predictor.joblib')
app = Flask(__name__)

@app.route('/')
def home():
    min_bmi = df['bmi'].min()
    max_bmi = df['bmi'].max()
    min_systolic = df['systolic'].min()
    max_systolic = df['systolic'].max()
    min_diastolic = df['diastolic'].min()
    max_diastolic = df['diastolic'].max()
    min_cholesterol = df['cholesterol'].min()
    max_cholesterol = df['cholesterol'].max()
    min_glucose = df['glucose'].min()
    max_glucose = df['glucose'].max()

    return render_template('diabetes_detector.html', min_bmi=min_bmi, max_bmi=max_bmi, min_systolic=min_systolic, 
                           max_systolic=max_systolic, min_diastolic=min_diastolic, max_diastolic=max_diastolic,
                           min_cholesterol=min_cholesterol, max_cholesterol=max_cholesterol, min_glucose=min_glucose,
                           max_glucose=max_glucose
                           )

@app.route('/inputs', methods=['GET', 'POST'])
def inputs():
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        bmi = request.form.get('bmi')
        bmi_category = request.form.get('bmi_category')
        systolic = request.form.get('systolic')
        diastolic = request.form.get('diastolic')
        bp_category = request.form.get('bp_category')
        cholesterol = request.form.get('cholesterol')
        cholesterol_category = request.form.get('cholesterol_category')
        glucose = request.form.get('glucose')

        data = pd.DataFrame({
            'age': [np.int64(age)],
            'gender': [gender],
            'bmi': [np.float64(bmi)],
            'bmi_category': [bmi_category],
            'systolic': [np.float64(systolic)],
            'diastolic': [np.float64(diastolic)],
            'blood_pressure_category': [bp_category],
            'cholesterol': [np.int64(cholesterol)],
            'cholesterol_category': [cholesterol_category],
            'glucose': [np.int64(glucose)]
        })
        
        prediction = model.predict(data)
        prediction = np.round(prediction[0]*100, 2)
        return render_template('diabetes_output.html', prediction=prediction)