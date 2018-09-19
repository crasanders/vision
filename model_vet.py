import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

vet_trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)
images = np.loadtxt('images.txt', dtype=str)
features = np.loadtxt('vet_resnet50.txt')

distances = squareform(pdist(features))

indices = {}
for i, im in enumerate(images):
    indices[im] = i

probs = np.zeros((vet_trials.shape[0], 3))

c = 1

for t, row in vet_trials.iterrows():
    target = indices[row['tarname']]
    distractor1 = indices[row['dist1name']]
    distractor2 = indices[row['dist2name']]

    trial_probs = np.zeros((3, 1))
    for i in range(1, 7):
        exemplar = indices[row['exemplar{}'.format(i)]]
        trial_probs[0] += np.exp(-c * distances[exemplar, target])
        trial_probs[1] += np.exp(-c * distances[exemplar, distractor1])
        trial_probs[2] += np.exp(-c * distances[exemplar, distractor2])

    trial_probs /= np.sum(trial_probs)
    probs[t] = trial_probs.reshape(3)

probs = pd.DataFrame(probs, columns=['target_response', 'dist1response', 'dist2response'])

model_response = pd.concat((vet_trials, probs), axis=1)

model_response.groupby(['cat', 'Task'])['target_response'].mean()
