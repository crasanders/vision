from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.callbacks import ModelCheckpoint
import numpy as np
from math import ceil

save_dir = 'Plane Classifiers'
npix = 224
imdir = 'Planes'
batch = 64
nepoch = 100

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=.1)

train_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix),
                                        subset='training', batch_size=batch)
val_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix),
                                      subset='validation', batch_size=batch)

hyperparms = [2, 256, -2.0286623630413607]
model = load_model('{}.hdf5'.format(hyperparms))

checkpoint = ModelCheckpoint('plane_classifier_classifier.hdf5'.format(hyperparms), save_best_only=True)

history = model.fit_generator(train_gen,
                              epochs=nepoch,
                              validation_data=val_gen,
                              steps_per_epoch=ceil(7353 / batch),
                              validation_steps=ceil(805 / batch),
                              callbacks=[checkpoint]
                              )

np.savetxt('pc_classifier_training_acc.txt', history.history['acc'])
np.savetxt('pc_classifier_validation_acc.txt', history.history['val_acc'])