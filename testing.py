from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy
from PIL import Image

img_width, img_height = 150, 150

train_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/train'
validation_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/validation'
nb_train_samples = 180
nb_validation_samples = 20
epochs = 50
batch_size = 10



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

img_width, img_height = 150, 150
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

img = numpy.array(Image.open('C:/Users/MASSRIDER/PycharmProjects/untitled4/img.jpg').resize((150,150))).reshape((1, 150, 150, 3))
img = img.astype('float32')
img/= 255.


prd = model.predict(img)
idx = int(numpy.round(prd[0, 0]))
label = ["sour", "sweet"][idx]
print(label)
print(prd)