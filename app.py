from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

# === Google Drive Large File Handler ===
def download_from_gdrive(file_id, destination):
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(base_url, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# === Step 1: Download model if not present ===
model_path = "final_random_forest_model.pkl"
file_id = "1etfH9HbBSGLb_cvxWgAsrZFP_MFsd480"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    download_from_gdrive(file_id, model_path)
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
