import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint

import numpy as np

npix = 224
imdir = 'Processed Planes'
batch = 64

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

ncat = 25
dropout = .5
hyperparms = [2, 256, -2.0286623630413607]
nlayers, nunits, loglr = hyperparms
lr = 10 ** loglr

base_model = DenseNet121(include_top=False, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

for lyr in range(nlayers):
    x = Dense(nunits, activation='relu', name='dense_{}'.format(lyr))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

x = Dense(ncat, activation='softmax', name='prediction')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['acc'])

checkpoint = ModelCheckpoint('plane_classifier.hdf5'.format(hyperparms), save_best_only=True)

result = model.fit_generator(train_gen,
                             epochs=200,
                             validation_data=val_gen,
                             callbacks=[checkpoint],
                             verbose=True)

np.savetxt('pc_classifier_training_acc.txt', result.history['acc'])
np.savetxt('pc_classifier_validation_acc.txt', result.history['val_acc'])

train = np.loadtxt('pc_classifier_training_acc.txt')
val = np.loadtxt('pc_classifier_validation_acc.txt')
