import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

simulations = pd.read_pickle('model_simulations.pkl')

d = simulations.z. \
    groupby(['Category', 'Layer', 'c'], as_index=False).mean()
m = d.groupby(['Category'], as_index=False)['det_acc'].idxmax()
best = d.iloc[m][['Category', 'Layer', 'c', 'det_acc', 'acc']]

# m = d.groupby(['Category', 'Layer'], as_index=False)['prob_acc'].max()
# sns.lineplot(x='Layer', y='prob_acc', hue='Category', data=m, ci=None)
# plt.show()
# plt.close()

#m = d.query('c < 5000').groupby(['Category', 'c'], as_index=False)['prob_acc'].max()
#sns.lineplot(x='c', y='prob_acc', hue='Category', data=m, ci=None)
#plt.show()
#plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
m = d.groupby(['Category'], as_index=False)['det_acc'].max()
p = sns.barplot(x='Category', y='det_acc', data=m, ci=None, ax=ax)
p.axhline(1/3, ls='--', color='black')
p.set_ylim((0, 1))
p.set_ylabel('Maximum Accuracy')
for j, patch in enumerate(p.axes.patches):
    p_x = patch.get_x() + patch.get_width() / 4
    p_y = .8
    plt.text(p_x, p_y, 'Layer' + str(list(best['Layer'])[j]))

plt.savefig('vet_category_accuracies.png', size=(1200, 600))
plt.show()
plt.close()

#c = best.iloc[0]['c']
#butterflies = simulations.query('Category == "Butterflies" and Layer == 15 and c == {}'.format(c))

# layers = dict(zip(best['Category'], best['Layer']))
#
# with open('best_layers.pkl', 'wb') as file:
#     pickle.dump(layers, file)
