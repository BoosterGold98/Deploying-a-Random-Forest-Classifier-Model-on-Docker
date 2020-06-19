# importing libraries
from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
swagger = Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "add apidocs after the url"

@app.route('/predict',methods=["Get"])
def predict_iris():
    
    """Uses Iris Dataset 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    s_length=request.args.get("sepal_length")
    s_width=request.args.get("sepal_width")
    p_length=request.args.get("petal_length")
    p_width=request.args.get("petal_width")
    prediction=classifier.predict([[s_length,s_width,p_length,p_width]])
    print(prediction)
    return "Hello The answer is "+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_iris_file():
    """Uses Iris Dataset 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)          # running host on 0.0.0.0 onport 8000