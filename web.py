import os
import keras
import numpy as np
import numpy
import keras

import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import ImageDataGenerator
from flask import Response
from flask import Flask, render_template, request, url_for
from camera import VideoCamera

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image
from werkzeug.utils import secure_filename




base_model = keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet', pooling='avg', input_shape=(160,160,3))
x = base_model.output
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(base_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
graph = tf.get_default_graph()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)

model.load_weights('C:/Users/MASSRIDER/PycharmProjects/untitled4/first_try.h5')



web = Flask(__name__)


UPLOAD_FOLDER = os.path.basename('upload')
web.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def predict(path):
    # img = Image.open(path).resize((160,160))
    # img = np.array(img).reshape((1,160,160,3))
    #
    # return model.predict(img)
    global graph
    with graph.as_default():
        img = Image.open(path).resize((160, 160))
        img = np.array(img).reshape((1, 160, 160, 3))
        img = img.astype('float32')
        img /= 127.5
        img -= 1
        return model.predict(img)

@web.route('/')
def upload():
    return render_template('index.html')



@web.route('/upupup', methods=['GET','POST'])
def uploads_files():
    if request.method == 'POST':
        file = request.files['image']
        print("hello")
        path = 'upload/img.jpg'
        file.save(path)
        prd = predict(path)
        idx = int(numpy.round(prd[0, 0]))
        label = ["sour", "sweet"][idx]
        print(label)
        return render_template('select.html',answer=label)
    else:
        return render_template('select.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        img = np.array([frame], dtype='float32')
        prd = model.predict(img)
        idx = int(numpy.round(prd[0, 0]))
        label = ["sour", "sweet"][idx]
        print(label)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@web.route('/camera')
def video_feed():
    # return render_template('camera.html')
    #
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
  web.run(debug=True)
  web.run(host='0.0.0.0', port=5000)
