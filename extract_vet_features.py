from keras.applications import vgg16, inception_resnet_v2
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import os

directory = 'VET'
nPixels = 224

X = []
vet_info = {}
i = 0
for subdir, dirs, files in os.walk(directory):
    dirs.sort()
    for file in sorted(files):
        if file.endswith(".bmp"):
            vet_info[file] = {}
            vet_info[file]['category'] = subdir.split('/')[1]
            vet_info[file]['index'] = i
            img = image.load_img(os.path.join(subdir, file), target_size=(nPixels, nPixels))
            x = image.img_to_array(img)
            X.append(x)
            i += 1
X = np.stack(X)

image_vectors = X[:, :, :, 0].reshape((i, -1))
np.savetxt('vet_pixels.txt', image_vectors, fmt='%.3i')

base_model = vgg16.VGG16(weights='imagenet', include_top=True, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
features = model.predict(vgg16.preprocess_input(X))
np.savetxt('vet_vgg16.txt', features, fmt='%.18f')

model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(inception_resnet_v2.preprocess_input(X))
np.savetxt('vet_inceptionresnetv2.txt', features, fmt='%.18f')
