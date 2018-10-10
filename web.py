# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# web = Flask(__name__)
#
#
#
#
#
# @web.route('/hello')
# def home():
#     return render_template('home.html')
#
#
# @web.route('/hello', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(secure_filename(f.filename))
#         return 'file uploaded successfully'
#
#
# if __name__ == '__main__':
#   web.run (debug=True)
#   web.run(host='0.0.0.0', port=5000)

###############################################################################################################################################
import os
import keras
import numpy as np
import numpy
from flask import Flask, render_template, request
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image
from werkzeug.utils import secure_filename

input_shape = (150,150,3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

model.load_weights('C:/Users/MASSRIDER/PycharmProjects/untitled4/first_try.h5')



web = Flask(__name__)


UPLOAD_FOLDER = os.path.basename('upload')
web.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def predict(path):
    img = Image.open(path).resize((150,150))
    img = np.array(img).reshape((1,150,150,3))

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
        prd = numpy.zeros((1,1))
        idx = int(numpy.round(prd[0, 0]))
        label = ["sour", "sweet"][idx]
        print(label)
        return render_template('home.html',answer=label)
    else:
        return render_template('home.html')




if __name__ == '__main__':
  web.run(debug=True)
  web.run(host='0.0.0.0', port=5000)





