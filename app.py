import joblib 
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd

app = Flask(__name__)

#load out pkl model
model = joblib.load(open('reg_model1.pkl','rb'))
columns = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

@app.route('/')
def home():
    return render_template('Home.html')

# @app.route('/predict_api',methods = ['POST'])
# def predict_api() :
#     data = request.json['data']
#     print(data)
#     new_data = np.array([data[i] for i in columns]).reshape(1,-1) 
#     output = model.predict(new_data)
#     final_output = np.expm1(output[0])
#     return jsonify(final_output)

@app.route('/predict',methods=['POST'])
def predict():
   
   data = request.json['data']
   final_input = np.array([float(data[i]) for i in columns]).reshape(1,-1)
   output = model.predict(final_input)[0]
   final_output = np.expm1(output)
   dollar_value = final_output * 100000

   return jsonify({'prediction': float(dollar_value)})

if __name__ =='__main__':
    app.run(debug=True)
    
