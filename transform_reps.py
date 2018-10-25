import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel
import os
import pandas as pd

directory = 'VET'
file_format = '.bmp'

trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)
distinct = trials.groupby(['Subcategory', 'tarname'], as_index=False).mean()[['Subcategory', 'tarname']]
d = {}
for i, row in distinct.iterrows():
    d[row['tarname']] = row['Subcategory']

labels = []
for subdir, dirs, files in os.walk(directory):
    dirs.sort()
    sorted_files = sorted(filter(lambda f: f.endswith(file_format), files))
    for file in sorted_files:
        if file in d:
            labels.append(d[file])
        else:
            labels.append('!')

categories = np.loadtxt('categories.txt', dtype=str)
encoded_labels = LabelEncoder().fit_transform(labels)

net = 'vgg16'
features = np.loadtxt('vet_{}.txt'.format(net))

for category in set(categories):
    X = features[categories == category]
    y = encoded_labels[categories == category]

    clf = LogisticRegressionCV(10, cv=LeaveOneOut(), penalty='l1', multi_class='ovr', solver='liblinear', n_jobs=-1)
    sfm = SelectFromModel(clf).fit(X, y)
    importance = sfm.get_support()

    features[categories == category] *= importance

np.savetxt('vet_transformed.txt', features)
