from ast import dump
from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('cats_and_dogs_small_1.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def pred():
    if(request.method == "POST"):
        # return request.form['image']
        image = tf.keras.preprocessing.image.load_img('test_images/' + request.form['image'])
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = tf.image.resize(input_arr, [150,150])
        input_arr = (input_arr)*1/255.

        input_arr = np.array([input_arr])

        prediction = model.predict(input_arr)

        if prediction >= 0.5:
            prediction = 1
            return "1 : Dog"
        else:
            prediction = 0
            return "0 : Cat"
        # return str(input_arr.shape)

if __name__ == '__main__':
    app.run(debug=True)