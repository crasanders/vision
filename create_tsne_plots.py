import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch

category_exemplars = pd.read_csv('category_exemplars.txt', delim_whitespace=True)
exemplar_ids = pd.read_csv('category_exemplars_ids.txt', delim_whitespace=True)


def plot_images(folder_path, save_folder, cat, coordinates, title, fig_size, im_zoom):
    n_categories = 6
    colors = sns.color_palette("hls", n_categories)
    legend_elements =[]
    for c, color in enumerate(colors):
        legend_elements.append(Patch(edgecolor=color, facecolor='none', label=category_exemplars[cat][c], linewidth=2))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(1, 1, 1)
    for subdir, dirs, files in os.walk(folder_path):
        for i, file in enumerate(sorted(files)):
            if file.endswith('.png'):
                file_name = file.split('.')[0]
                exemplar = False
                studied = False
                for j, e in enumerate(category_exemplars[cat]):
                    if e in file_name:
                        if file_name in list(exemplar_ids[cat]):
                            studied = True
                        exemplar = True
                        break
                image = plt.imread(os.path.join(subdir, file))
                im = OffsetImage(image, zoom=im_zoom)
                x0 = x[i]
                y0 = y[i]
                if studied:
                    ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True, pad=0.1,
                                        bboxprops=dict(edgecolor=colors[j], linestyle='-', linewidth=2))
                elif exemplar:
                    ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True, pad=0.1,
                                        bboxprops=dict(edgecolor=colors[j], linestyle='--', linewidth=2))
                else:
                    ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)

                ax.add_artist(ab)
                ax.update_datalim(np.column_stack([x0, y0]))
                ax.autoscale()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.axis('equal')
    plt.legend(handles=legend_elements, loc='best')
    plt.savefig(os.path.join(save_folder, title + '.png'))


categories = np.loadtxt('categories.txt', dtype=str)

rep = 'resnet50'
save_folder = 'tsne_plots'

X = np.loadtxt('vet_{}.txt'.format(rep))
tsne = TSNE().fit_transform(X)

plot = sns.scatterplot(tsne[:, 0], tsne[:, 1], hue=categories)
lgd = plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join(save_folder, 'vet_{}_tsne.png'.format(rep)), bbox_extra_artists=(lgd,), bbox_inches='tight')

for cat in set(categories):
    x = X[categories == cat, :]
    tsne = TSNE(perplexity=10).fit_transform(x)
    plot_images(os.path.join('VET_pngs', cat), 'tsne_plots', cat, tsne, 'vet_{}_{}_tsne'.format(rep, cat), 12, .1)

