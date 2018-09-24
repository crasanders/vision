import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import os

rep = 'vgg16'

vet_trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)
images = np.loadtxt('images.txt', dtype=str)
features = np.loadtxt('vet_{}.txt'.format(rep))

distances = pdist(features)
distances /= np.max(distances)
distances = squareform(distances)

indices = {}
for i, im in enumerate(images):
    indices[im] = i

n_exemplars = 6
n_choices = 3

model_responses = []
for c in np.arange(0, 100, 2):
    probs = np.zeros((vet_trials.shape[0], n_choices))

    for t, row in vet_trials.iterrows():
        target = indices[row['tarname']]
        distractor1 = indices[row['dist1name']]
        distractor2 = indices[row['dist2name']]
        choices = [target, distractor1, distractor2]

        trial_distances = np.zeros((n_exemplars, n_choices))

        for i in range(n_exemplars):
            exemplar = indices[row['exemplar{}'.format(i+1)]]
            for j in range(n_choices):
                trial_distances[i, j] = distances[exemplar, choices[j]]

        similarities = np.exp(-c * trial_distances)
        summed_similarities = np.sum(similarities, axis=0)
        trial_probs = summed_similarities / np.sum(summed_similarities)

        probs[t] = trial_probs.reshape(n_choices)

    probs = pd.DataFrame(probs, columns=['target_response', 'dist1response', 'dist2response'])

    model_response = pd.concat((vet_trials, probs), axis=1)
    model_response['c'] = c
    model_responses.append(model_response)

all_models = pd.concat(model_responses)

folder = 'accuracy_plots'

by_category = all_models.groupby(['Category', 'c'], as_index=False)['target_response'].mean()
plot = sns.lineplot(x='c', y='target_response', hue='Category', data=by_category)
lgd = plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Accuracy')
plt.savefig(os.path.join(folder, 'vet_{}_accuracy.jpg'.format(rep)), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

for cat in set(vet_trials['Category']):
    category = all_models.query('Category == "{}"'.format(cat))
    by_subcategory = category.groupby(['Subcategory', 'c'], as_index=False)['target_response'].mean()
    plot = sns.lineplot(x='c', y='target_response', hue='Subcategory', data=by_subcategory)
    lgd = plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(folder, 'vet_{}_{}_accuracy.jpg'.format(rep, cat)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()