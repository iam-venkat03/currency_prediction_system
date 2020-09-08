# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Krish Naik
"""


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='C:/Users/ADMIN/Desktop/model_30_epoch.h5'
MODEL_PATH2 = 'C:/Users/ADMIN/Desktop/country_35_epoch_final.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model_2 = load_model(MODEL_PATH2)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
   
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=preds.argmax()

    country = model_2.predict(x)
    country = country.argmax()

    curr_dict = {0:"1",1:"10",2:"100", 3:"1000",4:"2",5:"20", 6:"200",7:"2000",8:"5",9:"50",10:"500"}
    coun_dict = {0:"India", 1:"Singapore", 2:"USA"}

    final = str(curr_dict[preds]) + " , " + str(coun_dict[country])
    
    
    return final


@app.route('/', methods=['GET'])
def index(): 
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
