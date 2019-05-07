from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from math import ceil
import numpy as np

nepoch = 200
npix = 224
batch_size = 64
imdir = 'Planes'

base_model = ResNet50(include_top=False, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Dense(25, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model = multi_gpu_model(model, gpus=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=.1)

train_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix), subset='training',
                                        batch_size=batch_size)
val_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix), subset='validation',
                                      batch_size=batch_size)

checkpoint = ModelCheckpoint('plane_classifier.hdf5', save_best_only=True)

history = model.fit_generator(train_gen,
                              epochs=nepoch,
                              validation_data=val_gen,
                              steps_per_epoch=ceil(7353 / batch_size),
                              validation_steps=ceil(805 / batch_size),
                              callbacks=[checkpoint]
                              )

np.savetxt('plane_classifier_training_acc.txt', history.history['acc'])
np.savetxt('plane_classifier_validation_acc.txt', history.history['val_acc'])

#df = pd.DataFrame({'Epoch':list(range(1, nepoch+1)), 'Training': history.history['acc'], 'Validation': history.history['val_acc']})
#df = pd.melt(df, 'Epoch', ['Training', 'Validation'], var_name='Set', value_name='Accuracy')
#p = sns.lineplot(x='Epoch', y='Accuracy', hue='Set', data=df)
#plt.ylim(0, 1)
#plt.savefig('plane_classifier_accuracy.pdf')


