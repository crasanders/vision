import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

all_models = pd.read_csv('model_results.csv')
all_models['Accuracy'] = all_models['target_response']

folder = 'accuracy_plots'

mymax = all_models.query('Task == "Different"').groupby(
    ['c', 'Representation', 'Category', 'Subcategory', 'Cleaned'], as_index=False).mean().groupby(
    ['Representation', 'Category', 'Subcategory', 'Cleaned'], as_index=False)['Accuracy'].max()
sns.catplot('Category', 'Accuracy', col='Representation', hue='Cleaned', kind='bar', data=mymax, aspect=1.5, legend_out=True)
plt.savefig(os.path.join(folder, 'vet_maximimum_accuracy.jpg'))
# plt.show()
plt.close()

mymax.to_csv('model_results_max_accuracy_subcategories.csv')


by_network = all_models.query('Task == "Different"').groupby(
    ['c', 'Representation', 'Cleaned'], as_index=False)['Accuracy'].mean()
sns.lineplot('c', 'Accuracy', hue='Representation', style='Cleaned', data=by_network)
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(folder, 'vet_accuracy_overall.jpg'),
#             bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()
plt.close()

nrep = all_models['Representation'].nunique()
f, axes = plt.subplots(ncols=nrep, sharex=True, sharey=True, figsize=(16, 12))
for i, rep in enumerate(set(all_models['Representation'])):
    by_category = all_models.query('Task == "Different" and Representation == "{}"'.format(rep)).groupby(
        ['c', 'Category', 'Cleaned'], as_index=False)['Accuracy'].mean()
    if i == nrep-1:
        lgd = 'full'
    else:
        lgd = False
    sns.lineplot('c', 'Accuracy', hue='Category', style='Cleaned', data=by_category, ci=None, ax=axes[i], legend=lgd,
                 ).set_title('Network = {}'.format(rep))
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(folder, 'vet_accuracy_categories.jpg'),
            bbox_extra_artists=(legend,), bbox_inches='tight')
# plt.show()
plt.close()

for cat in set(all_models['Category']):
    category = all_models.query('Category == "{}" and Task == "Different"'.format(cat))

    f, axes = plt.subplots(ncols=nrep, sharex=True, sharey=True, figsize=(16, 12))
    for i, rep in enumerate(set(all_models['Representation'])):
        by_subcategory = category.query('Task == "Different" and Representation == "{}"'.format(rep)).groupby(
            ['c', 'Subcategory', 'Cleaned'], as_index=False)['Accuracy'].mean()
        if i == nrep-1:
            lgd = 'full'
        else:
            lgd = False
        sns.lineplot('c', 'Accuracy', hue='Subcategory', style='Cleaned', data=by_subcategory, ci=None, ax=axes[i],
                     legend=lgd).set_title('Network = {}'.format(rep))

    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    plt.savefig(os.path.join(folder, 'vet_accuracy_{}_subcategories.jpg'.format(cat)),
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()
