import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

vet_trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)
images = np.loadtxt('images.txt', dtype=str)

model_responses = []

for net in ['vgg16', 'inceptionv3', 'resnet50', 'densenet201']:
    for cleaned in [True, False]:
        rep = net
        if cleaned:
            rep = 'cleaned_' + net

        else:
            rep = 'raw_' + net

        features = np.loadtxt('vet_{}.txt'.format(rep))

        distances = pdist(features)
        distances = squareform(distances)
        distances /= np.max(distances)

        indices = {}
        for i, im in enumerate(images):
            indices[im] = i

        n_exemplars = 6
        n_choices = 3

        for log_c in np.arange(0, 3, .1):
            c = 10 ** log_c
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
                # trial_probs[0] = trial_probs[0] == np.max(trial_probs)

                probs[t] = trial_probs.reshape(n_choices)

            probs = pd.DataFrame(probs, columns=['target_response', 'dist1response', 'dist2response'])
            probs['Representation'] = net
            probs['Cleaned'] = cleaned

            model_response = pd.concat((vet_trials, probs), axis=1)
            model_response['log_c'] = log_c
            model_response['c'] = c
            model_responses.append(model_response)

all_models = pd.concat(model_responses)
all_models.to_csv('model_results.csv')

