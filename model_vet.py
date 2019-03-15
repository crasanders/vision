import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

vet_trials = pd.read_csv('vet_trials.csv').query('Task == "Different"')

categories = vet_trials['Category'].unique()
layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool', 'fc1', 'fc2', 'predictions']
metrics = ['euclidean', 'cityblock', 'cosine', 'jaccard']

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

def model_predictions(layer, metric, c, gamma, biases, memory):
    predictions = []
    for i, row in data.iterrows():
        trial_exemplars = []
        for j in range(1, 7):
            ex = row['exemplar{}'.format(j)]
            ex_features = features[ex][layer]
            trial_exemplars.append(ex_features)
        trial_exemplars = np.array(trial_exemplars)

        target = row['tarname']
        dist1 = row['dist1name']
        dist2 = row['dist2name']

        target_features = features[target][layer]
        dist1_features = features[dist1][layer]
        dist2_features = features[dist2][layer]

        # trial_choices = [dist1_features, dist2_features]
        # trial_choices.insert(row['tarloc'] - 1, target_features)
        trial_choices = [target_features, dist1_features, dist2_features]
        trial_choices = np.array(trial_choices)

        distances = cdist(trial_exemplars, trial_choices, metric=metric)
        sims = np.exp(-c * distances)
        weightedsims = (sims.T * memory).T
        sumsims = weightedsims.sum(axis=0)
        weightedsums = biases * sumsims ** gamma
        probs = weightedsums / weightedsums.sum()

        predictions.append(probs)
    return np.array(predictions)

def model_loss(space):
    # biases = np.array([space['b1'], space['b2'], 1])
    biases = np.array([1.] * 3)
    biases /= biases.sum()

    memory = np.array([space['v1'], space['v2'], space['v3'], space['v4'], space['v5'], 1])
    # memory = np.array([1.] * 6)
    memory /= memory.sum()

    predictions = model_predictions(space['layer'], space['metric'], space['c'], space['gamma'], biases, memory)

    responses = np.array(data.loc[:, 'resp1':'resp3'])
    nll = -np.sum(np.log(predictions) * responses)

    if not np.isnan(nll):
        status = STATUS_OK
    else:
        status = STATUS_FAIL

    mean_error = 1 - predictions[:, 0].mean()

    return {'loss': mean_error, 'status': status, 'predictions': predictions}


space = {
         'c': hp.loguniform('c', -2, 2),
         'gamma': hp.loguniform('gamma', -2, 2),
         'layer': hp.choice('layer', layers),
         'metric': hp.choice('metric', metrics)
        }

# for i in range(1, 3):
#     space['b{}'.format(i)] = hp.uniform('b{}'.format(i), 0, 10)

for i in range(1, 6):
    space['v{}'.format(i)] = hp.uniform('v{}'.format(i), 0, 10)

best_results = {}
for category in categories:
    data = vet_trials[vet_trials['Category'] == category]
    trials = Trials()
    best_parms = fmin(model_loss, space=space, algo=tpe.suggest, max_evals=10000, trials=trials)
    best_index = np.nanargmin(trials.losses())
    best_predictions = trials.results[best_index]['predictions']
    best_loss = trials.results[best_index]['loss']

    best_results[category] = {}
    best_results[category]['params'] = best_parms
    best_results[category]['predictions'] = best_predictions
    best_results[category]['loss'] = best_loss

    print(category, 1-best_loss)
