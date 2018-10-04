import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

all_models = pd.read_csv('model_results.csv')
all_models['Accuracy'] = all_models['target_response']

folder = 'accuracy_plots'

by_network = all_models.query('Task == "Different" and Category != "Planes"').groupby(
    ['log_c', 'Network'], as_index=False)['Accuracy'].mean()
sns.lineplot('log_c', 'Accuracy', hue='Network', data=by_network)
plt.show()

by_category = all_models.query('Task == "Different"').groupby(
    ['Category', 'log_c', 'Network'], as_index=False)['Accuracy'].mean()
plot = sns.FacetGrid(by_category, col='Network', height=4)
plot.map(sns.lineplot, 'log_c', 'Accuracy', ci=None)

plot = sns.FacetGrid(by_category, col='Network', hue='Category', height=4)
plot.map(sns.lineplot, 'log_c', 'Accuracy')

plot.add_legend()
plt.savefig(os.path.join(folder, 'vet_accuracy.jpg'))
# plt.show()
plt.close()

for cat in set(all_models['Category']):
    category = all_models.query('Category == "{}" and Task == "Different"'.format(cat))
    by_subcategory = category.groupby(['Subcategory', 'log_c', 'Network'], as_index=False)['Accuracy'].mean()

    plot = sns.FacetGrid(by_subcategory, col='Network', hue='Subcategory', height=4,
                         hue_order=sorted(set(category['Subcategory'])))
    plot.map(sns.lineplot, 'log_c', 'Accuracy')
    plot.add_legend()
    # plt.show()
    plt.savefig(os.path.join(folder, 'vet_{}_accuracy.jpg'.format(cat)))
    plt.close()
