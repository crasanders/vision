from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from os.path import join
import numpy as np


def data():
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

    return train_gen, val_gen


def create_model(train_gen, val_gen):
    save_dir = 'Plane Classifiers'
    ncat = 25

    nlayers = {{choice([0, 1, 2, 3, 4])}}
    nunits = {{choice([64, 128, 256, 512, 1024])}}
    dropout = {{uniform(.2, .8)}}
    loglr = {{uniform(-5, -1)}}
    lr = 10 ** loglr

    hyperparms = [nlayers, nunits, dropout, loglr]

    base_model = DenseNet121(include_top=False, pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    for lyr in range(nlayers):
        x = Dense(nunits, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    x = Dense(ncat, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['acc'])

    checkpoint = ModelCheckpoint(join(save_dir, '{}.hdf5'.format(hyperparms)), save_best_only=True)

    result = model.fit_generator(train_gen,
                                 epochs=50,
                                 validation_data=val_gen,
                                 callbacks=[checkpoint],
                                 verbose=True)

    validation_acc = np.amax(result.history['val_acc'])
    print(validation_acc, hyperparms)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                                data=data,
                                                algo=tpe.suggest,
                                                max_evals=25,
                                                trials=Trials())

    save_dir = 'Plane Classifiers'
    best_model.save(join(save_dir, 'best_model.hdf5'))
