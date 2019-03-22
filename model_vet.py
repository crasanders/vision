import pickle
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

with open('resnet50_features_cleaned_scaled.pkl', 'rb') as file:
    clean_features = pickle.load(file)

with open('resnet50_features_scaled.pkl', 'rb') as file:
    unclean_features = pickle.load(file)

features = {True: unclean_features, False: clean_features}


def model(row, layer, metric, c, background, aggregation):
    trial_exemplars = []
    for j in range(1, 7):
        ex = row['exemplar{}'.format(j)]
        ex_features = features[background][ex][layer]
        trial_exemplars.append(ex_features)
    trial_exemplars = np.array(trial_exemplars)

    if aggregation == 'none':
        pass
    if aggregation == 'mean':
        trial_exemplars = trial_exemplars.mean(axis=0).reshape(1, -1)
    if aggregation == 'sum':
        trial_exemplars = trial_exemplars.sum(axis=0).reshape(1, -1)
    if aggregation == 'max':
        trial_exemplars = trial_exemplars.max(axis=0).reshape(1, -1)

    target = row['tarname']
    dist1 = row['dist1name']
    dist2 = row['dist2name']

    tarloc = row['tarloc'] - 1

    target_features = features[background][target][layer]
    dist1_features = features[background][dist1][layer]
    dist2_features = features[background][dist2][layer]

    trial_choices = [dist1_features, dist2_features]
    trial_choices.insert(tarloc, target_features)

    distances = cdist(trial_exemplars, trial_choices, metric=metric)
    # sims = np.exp(-c * distances)
    # sumsims = sims.sum(axis=0)
    # probs = sumsims / sumsims.sum()

    scaled_dist = distances * -c
    probs = np.exp(logsumexp(scaled_dist, axis=0) - logsumexp(scaled_dist))

    responses = np.array(row['resp1':'resp3'])
    # nll = -np.sum(np.log(probs) * responses)
    acc = probs[tarloc]
    det_acc = np.argmax(probs) == tarloc

    result = dict(row)
    result.update(
        {'Distance{},{}'.format(i + 1, j + 1): distances[i][j] for i in range(distances.shape[0]) for j in range(3)})
    # result.update({'Similarity{},{}'.format(i + 1, j + 1): sims[i][j] for i in range(sims.shape[0]) for j in range(3)})
    # result.update({'SumSim{}'.format(i + 1): sumsims[i] for i in range(3)})
    result.update({'Prob{}'.format(i + 1): probs[i] for i in range(3)})
    # result.update({'nll': nll, 'prob_acc': acc, 'det_acc': det_acc})
    result.update({'prob_acc': acc, 'det_acc': det_acc})
    result.update({'Layer': layer, 'Metric': metric, 'Aggregation': aggregation, 'c': c, 'log_c': np.log10(c)})
    result.update({'Background': background})

    return result


vet_trials = pd.read_csv('vet_trials.csv').query('Task == "Different"')

layers = range(50)
metrics = ['euclidean', 'cityblock', 'cosine', 'jaccard']
aggregations = ['none', 'mean', 'max']
cs = np.logspace(0, 3, 20)
backgrounds = [True, False]

nrows = len(layers) * len(metrics) * len(cs) * len(backgrounds) * len(aggregations) * len(vet_trials)

results = []
for layer, metric, c, background, aggregation in itertools.product(layers, metrics, cs, backgrounds, aggregations):
    print(layer, metric, c, background, aggregation)
    for i, row in vet_trials.iterrows():
        result = model(row, layer, metric, c, background, aggregation)
        results.append(result)

results = pd.DataFrame(results)
for col in results.columns:
    if results[col].dtype == 'object':
        results[col] = results[col].astype('category')

results.to_pickle('model_simulations.pkl')

# results.to_csv('model_simulations.csv', index=False)
