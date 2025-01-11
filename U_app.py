from __future__ import division, print_function

# coding=utf-8
import os
import numpy as np

# Keras

from tensorflow import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input


from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='C:/Users/venka/Downloads/currency_prediction_system-master/currency_prediction_system-master/initial_currency_model.h5'
MODEL_PATH2 = 'C:/Users/venka/Downloads/currency_prediction_system-master/currency_prediction_system-master/initial_country_model.h5'

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
    preds= preds.argmax()

    country = model_2.predict(x)
    country = country.argmax()

    #curr_dict = {'1': 0,'1-2': 1,'1-4': 2,'10': 3,'100': 4,'1000': 5,'10000': 6,'20': 7,'200': 8,'5': 9,'50': 10,'500': 11,'5000': 12}
    #coun_dict = {0:" India ", 1:" Singapore ", 2:"United States of America"}

    coun_decision = {0: "Australia", 1: "Europe", 2:"Japan", 3: "Kuwait", 4: "Mexico", 5:"New Zealand", 6: "Switzerland", 7: "UK"}
    curr_dict = {0: '1', 1: '1-2', 2: '1-4', 3: '10', 4: '100', 5: '1000', 6: '10000', 7: '20', 8: '200', 9: '5', 10: '50', 11: '500', 12: '5000'}



    final = str(curr_dict[preds]) + " " + str(coun_decision[country])
    
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
