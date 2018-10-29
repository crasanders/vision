import numpy as np
import pickle
import pandas as pd
from metric_learn.mmc import MMC
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

with open('file_indices.pkl', 'rb') as f:
    indices = pickle.load(f)

trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)

sim1 = []
sim2 = []
dis1 = []
dis2 = []
simcats = []
discats = []

for i, row in trials.drop_duplicates(['tarname', 'dist1name', 'dist2name']).iterrows():
    dis1.append(indices[row['tarname']])
    dis2.append(indices[row['dist1name']])

    dis1.append(indices[row['tarname']])
    dis2.append(indices[row['dist2name']])

    discats.extend([row['Category']]*2)

for i, row in trials.drop_duplicates(['tarname',
                                      'exemplar1', 'exemplar2', 'exemplar3',
                                      'exemplar4', 'exemplar5', 'exemplar6']).iterrows():
    sim1.extend([indices[row['tarname']]] * 6)
    sim2.extend([indices[row['exemplar{}'.format(i+1)]] for i in range(6)])

    simcats.extend([row['Category']] * 6)

categories = np.loadtxt('categories.txt', dtype=str)

net = 'vgg16'
features = np.loadtxt('vet_{}.txt'.format(net))
features = scale(features)
pca = PCA()
pca_features = pca.fit_transform(features)
explained = pca.explained_variance_ratio_.cumsum()
features = pca_features[:, explained <= .95]


sim1 = np.array(sim1)
sim2 = np.array(sim2)
simcats = np.array(simcats)

mask = sim1 != sim2
sim1 = sim1[mask]
sim2 = sim2[mask]
simcats = simcats[mask]

dis1 = np.array(dis1)
dis2 = np.array(dis2)
discats = np.array(discats)

mmc = MMC(verbose=True)
mmc.fit(features, (sim1, sim2, dis1, dis2))
transformed = mmc.transform()
np.savetxt('vet_transformed_all.txt', transformed, fmt='%.18f')

offset = 0
for category in sorted(set(categories)):
    X = features[categories == category]

    a = sim1[simcats == category] - offset
    b = sim2[simcats == category] - offset
    c = dis1[discats == category] - offset
    d = dis2[discats == category] - offset

    mmc.fit(X, (a, b, c, d))

    features[categories == category] = mmc.transform()

    offset += X.shape[0]

np.savetxt('vet_transformed_ind.txt', features, fmt='%.18f')