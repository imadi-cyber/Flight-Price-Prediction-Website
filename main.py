import numpy as np
import pandas as pd
import flask
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model2 = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    lst = [int(i) for i in request.form.values()]
    final_features = pd.DataFrame({"airline" : [lst[0]], "source_city" : [lst[1]], "departure_time" : [lst[2]], "stops" : [lst[3]], "arrival_time" : [lst[4]], "destination_city" : [lst[5]], "class" : [lst[6]]})
    prediction = model2.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Expected fare would be Rs. {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model2.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
