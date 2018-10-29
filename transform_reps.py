import numpy as np
import pickle
import pandas as pd
from metric_learn.mmc import MMC
from sklearn.preprocessing import scale

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

    discats.extend([row['Category']] * 2)

    for j in range(6):
        dis1.append(indices[row['dist1name']])
        dis2.append(indices[row['exemplar{}'.format(j+1)]])

        dis1.append(indices[row['dist2name']])
        dis2.append(indices[row['exemplar{}'.format(j+1)]])

        discats.extend([row['Category']]*2)

for i, row in trials.drop_duplicates(['tarname',
                                      'exemplar1', 'exemplar2', 'exemplar3',
                                      'exemplar4', 'exemplar5', 'exemplar6']).iterrows():
    sim1.append(indices[row['tarname']])
    sim2.append(indices[row['exemplar{}'.format(row['tarnb'])]])

    simcats.append(row['Category'])

categories = np.loadtxt('categories.txt', dtype=str)

net = 'vgg16'
features = np.loadtxt('vet_{}.txt'.format(net))
features = scale(features)

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

mmc = MMC(verbose=True, diagonal=True)
offset = 0
for category in sorted(set(categories)):
    X = features[categories == category]

    a = sim1[simcats == category] - offset
    b = sim2[simcats == category] - offset
    c = dis1[discats == category] - offset
    d = dis2[discats == category] - offset

    features[categories == category] = mmc.fit_transform(X, (a, b, c, d))

    offset += X.shape[0]

np.savetxt('vet_mmc_ind_pca.txt', features, fmt='%.18f')