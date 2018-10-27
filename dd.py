import numpy as np
import numpy

import cv2

import numpy as np

import keras

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator


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

class VideoCamera(object):
    cap = cv2.VideoCapture(0)

    def __init__(self):
        while(True):
    # Capture frame-by-frame
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (160, 160))
    # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            img = np.array([frame])
            prd = model.predict(img)

            idx = int(numpy.round(prd[0, 0]))
            label = ["sour", "sweet"][idx]
            print(label)
# When everything done, release thqqe capture



    def __del__(self):
        self.cap.release()

