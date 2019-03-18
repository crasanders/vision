import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.applications import resnet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
import keras.backend as K
import numpy as np
import pickle

directory = 'VET Raw'
nPixels = 224

X = []
images = []
for subdir, dirs, files in os.walk(directory):
    dirs.sort()
    for file in sorted(files):
        if file.endswith(".bmp"):
            images.append(file)
            img = image.load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
            x = image.img_to_array(img)
            X.append(x)
X = np.stack(X)
X = resnet50.preprocess_input(X)

feature_sets = {}
layers = ['activation_{}'.format(i) for i in range(1, 50)] + ['fc1000']

for layer in layers:
    feature_sets[layer] = {}
    base_model = resnet50.ResNet50()

    x = base_model.get_layer(layer).output

    if layer != 'fc1000':
        x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)
    feature_sets[layer]= model.predict(X)

    K.clear_session() #just to be safe

image_features = {}
for i, image in enumerate(images):
    image_features[image] = {}
    for j, layer in enumerate(layers):
        image_features[image][j] = feature_sets[layer][i]

with open('resnet50_features.pkl', 'wb') as file:
    pickle.dump(image_features, file)



