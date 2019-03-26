import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.optimize import minimize

vet_trials = pd.read_csv('vet_trials.csv')
categories = np.loadtxt('categories.txt', dtype=str)
cats = sorted(np.unique(categories))
images = np.loadtxt('images.txt', dtype=str)

vet_trials = pd.read_csv('vet_trials.csv')
data = vet_trials.query('Task == "Different"')

mds = []
for category in cats:
    cat_mds = np.loadtxt('vet_{}_vgg16_mds_20.txt'.format(category), delimiter=',')
    mds.append(cat_mds)
mds = np.concatenate(mds)

dimminusone = mds.shape[1] - 1

indices = {}
for i, im in enumerate(images):
    indices[im] = i


def model(parms):
    offset = 0
    features = []
    for category in cats:
        w = np.append(parms[offset:offset + dimminusone], 1)
        features.append(mds[categories == category] * w)
        offset += dimminusone
    features = np.concatenate(features)

    nll = 0
    acc = []
    probabilities = []
    for i, row in data.iterrows():
        trial_exemplars = []
        for j in range(1, 7):
            ex = indices[row['exemplar{}'.format(j)]]
            ex_features = features[ex]
            trial_exemplars.append(ex_features)
        trial_exemplars = np.array(trial_exemplars)

        target = indices[row['tarname']]
        dist1 = indices[row['dist1name']]
        dist2 = indices[row['dist2name']]

        tarloc = row['tarloc'] - 1

        target_features = features[target]
        dist1_features = features[dist1]
        dist2_features = features[dist2]

        trial_choices = [dist1_features, dist2_features]
        trial_choices.insert(tarloc, target_features)

        distances = cdist(trial_exemplars, trial_choices)
        probs = np.exp(logsumexp(distances, axis=0) - logsumexp(distances))
        probabilities.append(probs)

        targets = np.array(row['resp1':'resp3'])
        # targets = [0, 0]
        # targets.insert(tarloc, 1)
        nll -= np.sum(np.log(probs) * np.array(targets))

        acc.append(probs[row['highest_resp']-1])
        # acc.append(probs[tarloc])

    probabilities = np.array(probabilities)
    acc = np.array(acc)
    return nll, probabilities, acc, acc.mean()


fit = minimize(lambda x: model(x)[0], [1]*dimminusone*len(cats), bounds=[(0, None)]*dimminusone*len(cats))
print(model(fit.x)[-1])

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.lineplot(x='c', y='prob_acc', ci=None, data=results)
# plt.show()
# plt.close()