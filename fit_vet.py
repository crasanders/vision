import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.optimize import minimize

vet_trials = pd.read_csv('vet_trials.csv')

for category in sorted(vet_trials['Category'].unique()):
    data = vet_trials.query('Task == "Different" and Category == "{}"'.format(category))
    mds = np.loadtxt('vet_{}_vgg16_mds.txt'.format(category), delimiter=',')
    mds /= mds.max()

    directory = 'VET Cleaned'
    features = {}
    i = 0
    for subdir, dirs, files in os.walk(directory):
        dirs.sort()
        for file in sorted(files):
            if file.endswith(".bmp") and subdir.split('/')[1] == category:
                    features[file] = mds[i]
                    i += 1


    def model(parms):
        w = np.append(parms, 1)

        nll = 0
        acc = []
        for i, row in data.iterrows():
            trial_exemplars = []
            for j in range(1, 7):
                ex = row['exemplar{}'.format(j)]
                ex_features = features[ex] * w
                trial_exemplars.append(ex_features)
            trial_exemplars = np.array(trial_exemplars)

            target = row['tarname']
            dist1 = row['dist1name']
            dist2 = row['dist2name']

            tarloc = row['tarloc'] - 1

            target_features = features[target] * w
            dist1_features = features[dist1] * w
            dist2_features = features[dist2] * w

            trial_choices = [dist1_features, dist2_features]
            trial_choices.insert(tarloc, target_features)

            distances = cdist(trial_exemplars, trial_choices)
            sims = np.exp(-distances)
            sumsims = sims.sum(axis=0)
            probs = sumsims / sumsims.sum()

            # targets = [0, 0]
            # targets.insert(tarloc, 1)
            targets = np.array(row['resp1':'resp3'])
            nll -= np.sum(np.log(probs) * np.array(targets))

            acc.append(probs[row['highest_resp']-1])

            return nll, np.array(acc).mean()

    fit = minimize(lambda x: model(x)[0], [1]*(7), bounds=[(0, None)]*(7))
    print(category, model(fit.x))

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.lineplot(x='c', y='prob_acc', ci=None, data=results)
# plt.show()
# plt.close()