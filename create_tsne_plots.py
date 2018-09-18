import numpy as np
from sklearn.manifold.t_sne import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

categories = np.loadtxt('categories.txt', dtype=str)

rep = 'densenet201'

X = np.loadtxt('vet_{}.txt'.format(rep))
tsne = TSNE().fit_transform(X)

plot = sns.scatterplot(tsne[:,0], tsne[:,1], hue=categories)
plt.savefig('vet_{}_tsne.jpg'.format(rep))
