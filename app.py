from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests
import gzip

app = Flask(__name__)

# === Step 1: Download compressed Gradient Boosting model if not present ===
model_url = "https://www.dropbox.com/scl/fi/txf9lrr74gs7tqu4jtrex/final_gb_model_compressed.pkl.gz?rlkey=l5rs7qjq9gqextg45v1j20hmu&st=v2a1hxpl&dl=1"
model_path = "final_gb_model_compressed.pkl.gz"

if not os.path.exists(model_path):
    print("Downloading compressed GB model from Dropbox...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# === Step 2: Define expected input features for GB model
FEATURES = ['OCCP', 'WKHP', 'SCHL', 'AGEP', 'SEX']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the Gradient Boosting model
        with gzip.open(model_path, "rb") as f:
            gb_model = pickle.load(f)

        # Collect and prepare input data
        data = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(data).reshape(1, -1)

        # Make prediction
        pred = gb_model.predict(input_array)[0]

        def interpret(pred):
            return "Above 50K" if pred == 1 else "50K or Below"

        return render_template(
            'result.html',
            input_values=dict(zip(FEATURES, data)),
            result_rf=interpret(pred)
        )

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=81)
