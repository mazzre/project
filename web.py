# # from flask import Flask, render_template, request
# # from werkzeug.utils import secure_filename
# # web = Flask(__name__)
# #
# #
# #
# #
# #
# # @web.route('/hello')
# # def home():
# #     return render_template('home.html')
# #
# #
# # @web.route('/hello', methods=['GET', 'POST'])
# # def upload_file():
# #     if request.method == 'POST':
# #         f = request.files['file']
# #         f.save(secure_filename(f.filename))
# #         return 'file uploaded successfully'
# #
# #
# # if __name__ == '__main__':
# #   web.run (debug=True)
# #   web.run(host='0.0.0.0', port=5000)
#
# ###############################################################################################################################################
import os
import keras
import numpy as np
import numpy
import keras
from flask import Flask, render_template, request
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

input_shape = (160,160,3)

base_model = keras.applications.mobilenet.MobileNet(include_top=True, weights = 'imagenet', pooling='avg', input_shape=(160,160,3))
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
    global graph
    with graph.as_default():
        img = Image.open(path).resize((160,160))
        img = np.array(img).reshape((1,160,160,3))
        return model.predict(img)


@web.route('/')
def upload():
    return render_template('home.html')

# @web.route('/upload', methods=['POST'])
# def upload_file():
#
#     file = request.files['image']
#     f = os.path.join(web.config['UPLOAD_FOLDER'], file.filename)
#     file.save(f)
#
#     return render_template('home.html')

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
        return render_template('home.html',answer=label)
    else:
        return render_template('home.html')




if __name__ == '__main__':
  web.run(debug=True)
  web.run(host='0.0.0.0', port=5000)





