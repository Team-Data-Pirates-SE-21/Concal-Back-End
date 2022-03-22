from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('concrete-predicter.joblib')  # model trained


@app.route('/', methods=['POST'])  # using post method to data send
def feedModel():
    newdata = request.form.get('inputs')  # getting data in json format
    print(newdata)
    newDataArr = newdata.split()  # split data into an array
    data = [
        [float(newDataArr[0]), float(newDataArr[1]), float(newDataArr[2]), float(newDataArr[3]), float(newDataArr[4]),
         float(newDataArr[5]), float(newDataArr[6]), float(newDataArr[7])]]  # adjusted array into 2d array
    prediction = np.array2string(model.predict(data)[0])  # do the prediction and get the output as a string
    print(prediction)
    return jsonify({
        "output": str(prediction)  # return the value within a dictionary json object
    })


if __name__ == '__main__':
    app.run(debug=True)
