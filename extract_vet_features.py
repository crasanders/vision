from keras.applications import vgg16, inception_v3, resnet50, densenet
from keras.preprocessing import image
from keras.models import Model
import keras.backend as K
import numpy as np
import os

directory = 'VET'
nPixels = 224

X = []
categories = []
images = []
for subdir, dirs, files in os.walk(directory):
    dirs.sort()
    for file in sorted(files):
        if file.endswith(".bmp"):
            images.append(file)
            categories.append(subdir.split('/')[1])
            img = image.load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
            x = image.img_to_array(img)
            X.append(x)
X = np.stack(X)

np.savetxt('categories.txt', categories, fmt='%s')
np.savetxt('images.txt', images, fmt='%s')

image_vectors = X[:, :, :, 0].reshape((X.shape[0], -1))
np.savetxt('vet_pixels.txt', image_vectors, fmt='%.3i')

base_model = vgg16.VGG16(weights='imagenet', include_top=True, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
features = model.predict(vgg16.preprocess_input(X.copy()))
np.savetxt('vet_vgg16.txt', features, fmt='%.18f')
K.clear_session()

model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(inception_v3.preprocess_input(X.copy()))
np.savetxt('vet_inceptionv3.txt', features, fmt='%.18f')
K.clear_session()

model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(resnet50.preprocess_input(X.copy()))
np.savetxt('vet_resnet50.txt', features, fmt='%.18f')
K.clear_session()

model = densenet.DenseNet201(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(densenet.preprocess_input(X.copy()))
np.savetxt('vet_densenet201.txt', features, fmt='%.18f')
K.clear_session()
