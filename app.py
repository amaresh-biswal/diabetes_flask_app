from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        data = [float(request.form[key]) for key in request.form]
        data_np = np.array(data).reshape(1, -1)

        # Scale the input
        scaled_data = scaler.transform(data_np)

        # Make prediction
        prediction = model.predict(scaled_data)[0]

        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
        return render_template('index.html', prediction_text=f'Result: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
