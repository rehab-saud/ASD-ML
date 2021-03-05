from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('Toddler_Autism_dataset.csv')

x = dataset.iloc[:, 1:11].values

sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(x)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def index():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(sc.transform(final_features))
    
    if prediction == 1:
        pred = "The score indicates, the child has ASD."
    elif prediction == 0:
        pred = "The score indicates, the child has no ASD."
    output = pred
    
    return render_template('prediction.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    