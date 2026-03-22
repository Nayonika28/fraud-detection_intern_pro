from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Fraud Detection Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']

    prediction = model.predict([data])[0]
    probability = model.predict_proba([data])[0][1]

    return jsonify({
        "fraud": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)