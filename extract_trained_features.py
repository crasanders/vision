from keras.models import load_model, Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
import os
import numpy as np

def extract_trained_plane_features(directory, label):
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
    X = preprocess_input(X)

    base_model = load_model('plane_classifier.hdf5')
    x = base_model.get_layer(label).output

    model = Model(inputs=base_model.input, outputs=x)
    pred = model.predict(X)

    np.savetxt(os.path.join('trained_plane_features', 'trained_plane_layer_{}.txt'.format(label)), pred, fmt='%.18f')


extract_trained_plane_features('/home/sandeca1/vision/VET Raw/Planes', 'dense_0')
extract_trained_plane_features('/home/sandeca1/vision/VET Raw/Planes', 'dense_1')
extract_trained_plane_features('/home/sandeca1/vision/VET Raw/Planes', 'prediction')

