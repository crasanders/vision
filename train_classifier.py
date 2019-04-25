from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import multi_gpu_model

npix = 224
imdir = 'Processed Planes'

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

train_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix), subset='training')
val_gen = datagen.flow_from_directory(imdir, class_mode='categorical', target_size=(npix, npix), subset='validation')

history = model.fit_generator(train_gen,
                              epochs=10,
                              validation_data=val_gen)
