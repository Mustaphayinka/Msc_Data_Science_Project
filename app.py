from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

# === Step 1: Download model from Dropbox if not present ===
model_url = "https://www.dropbox.com/scl/fi/iakt7i51zl42yqfq5tehx/final_random_forest_model.pkl?rlkey=fybb3jozjw2mk2ccbcqnwmj9w&st=widp1j0e&dl=1"
model_path = "final_random_forest_model.pkl"

if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# === Step 2: Load model ===
with open(model_path, "rb") as f:
    try:
        rf_model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Make sure the downloaded file is a valid .pkl file.\n{e}")

# === Step 3: Define features ===
FEATURES = ['OCCP', 'AGEP', 'POBP', 'WKHP', 'SCHL']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(data).reshape(1, -1)

        # Make prediction using Random Forest model
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
