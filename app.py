
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('concrete-predicter.joblib')


@app.route('/', methods=['POST'])
def feedModel():
    data = request.get_json()
    prediction = np.array2string(model.predict(data)[0])

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)  # change to '0.0.0.0' your IPv4 address
