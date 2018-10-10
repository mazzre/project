from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras


img_width, img_height = 150, 150

train_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/train'
validation_data_dir = 'C:/Users/MASSRIDER/PycharmProjects/untitled4/validation'
nb_train_samples = 180
nb_validation_samples = 20
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



base_model = keras.applications.resnet50.ResNet50(include_top = False,weights ='imagenet',pooling = 'avg')
x = base_model.output
x = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.models.Model(base_model.input, x)


model.compile(keras.optimizers.sgd(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['acc'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,
    width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(192,192),
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


print(history)
print (history.history['val_acc'][-1])