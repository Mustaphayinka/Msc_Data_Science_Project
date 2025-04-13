from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests
import gzip

app = Flask(__name__)

# === Step 1: Download compressed model from Dropbox if not present ===
model_url = "https://www.dropbox.com/scl/fi/xupzjzfx5q6fghlx70ncf/final_rf_model_compressed.pkl.gz?rlkey=fenxcl97wu8njtq34tavybziq&st=s3f59r64&dl=1"
model_path = "final_rf_model_compressed.pkl.gz"

if not os.path.exists(model_path):
    print("Downloading compressed model from Dropbox...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# === Step 2: Load compressed model ===
with gzip.open(model_path, "rb") as f:
    try:
        rf_model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Make sure the downloaded file is a valid compressed .pkl file.\n{e}")

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