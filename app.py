# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:34:53 2022
@author: pratiksanghvi
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sklearn
import joblib
import pickle


# load the model from disk
filename = 'nlp_model.pkl'
nbInitialize = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transformTheMessageToVector.pkl','rb'))
app = Flask(__name__)

# First time whne you click on the link the home page will be opened
@app.route('/')
def home():
	return render_template('home.html')

# Once you click on Predict button then results page will show
@app.route('/predict',methods=['POST'])
def predict():
        
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        transform_incoming_Message_to_Vector = cv.transform(data).toarray()
        my_prediction = nbInitialize.predict(transform_incoming_Message_to_Vector)
    
    return render_template('result.html',prediction = my_prediction)      
        
        
if __name__ == '__main__':
	app.run(debug=True)        
        
  
