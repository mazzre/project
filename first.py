from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
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
    vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('C:/Users/MASSRIDER/PycharmProjects/untitled4/first_try.h5')
print(history)
print(history.history['val_acc'][-1])