from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy
from PIL import Image
import keras

img_width, img_height = 160,160

train_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/train'
validation_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/validation'
nb_train_samples = 180
nb_validation_samples = 20
epochs = 5
batch_size = 16



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



base_model = keras.applications.mobilenet.MobileNet(include_top=True, weights = 'imagenet', pooling='avg', input_shape=(160,160,3))
x = base_model.output
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(base_model.input, x)


model.compile(keras.optimizers.sgd(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['acc'])


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

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

img = numpy.array(Image.open('C:/Users/MASSRIDER/PycharmProjects/untitled4/upload/img.jpg').resize((160,160))).reshape((1, 160,160, 3))
img = img.astype('float32')
img/= 255.


prd = model.predict(img)
idx = int(numpy.round(prd[0, 0]))
label = ["sour", "sweet"][idx]
print(label)
print(prd)