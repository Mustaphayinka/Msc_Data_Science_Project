from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the final Random Forest model
rf_model = joblib.load("final_random_forest_model.pkl")

# Expected input features based on RF
FEATURES = ['OCCP', 'AGEP', 'POBP', 'WKHP', 'SCHL']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(data).reshape(1, -1)

        # Make prediction using RF model
        pred_rf = rf_model.predict(input_array)[0]

        def interpret(pred):
            return "Above 50K" if pred == 1 else "50K or Below"

        return render_template(
            'result.html',
            input_values=dict(zip(FEATURES, data)),
            result_rf=interpret(pred_rf)
        )
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
