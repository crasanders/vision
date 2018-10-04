import pandas as pd
import os
import matplotlib.pyplot as plt

vet_trials = pd.read_csv('vet_trials.txt', delim_whitespace=True)

trials = vet_trials.query('Subcategory == "StearmanPT13"')

f, subplots = plt.subplots(len(trials), 3, figsize=(12, 12))
plt.tight_layout()

j = 0
for i, row in trials.iterrows():
    subdir = os.path.join('VET_pngs', row['Category'])

    file = row['tarname'].split('.')[0]
    image = plt.imread(os.path.join(subdir, file + '.png'))
    subplots[j, 0].imshow(image)
    subplots[j, 0].set_title(file)
    subplots[j, 0].axis('off')

    file = row['dist1name'].split('.')[0]
    image = plt.imread(os.path.join(subdir, file + '.png'))
    subplots[j, 1].imshow(image)
    subplots[j, 1].set_title(file)
    subplots[j, 1].axis('off')

    file = row['dist2name'].split('.')[0]
    image = plt.imread(os.path.join(subdir, file + '.png'))
    subplots[j, 2].imshow(image)
    subplots[j, 2].set_title(file)
    subplots[j, 2].axis('off')

    j += 1

plt.savefig('trial_plot.png', bbox_inches='tight')
