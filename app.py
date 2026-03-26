import joblib 
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np 
import pandas as pd

app = Flask(__name__)


#load out pkl model
model = joblib.load(open('reg_model1.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api() :
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    
    output = model.predict(new_data)
    final_output = np.expm1(output[0])
    return jsonify(final_output)
    
if __name__ =='__main__':
    app.run(debug=True)

