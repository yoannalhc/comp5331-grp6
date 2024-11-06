import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from os.path import join

def plot_cluster_result(pair1, pair2, label1, label2):

    # Compute PCA of data set
    seed = 0
    pca1 = PCA(n_components=pair1.shape[1], random_state=seed)
    pca1_arr = pca1.fit_transform(pair1)
    pair1_pca = pd.DataFrame(pca1_arr,
                         columns=['PC%i' % (ii + 1) for ii in range(pca1_arr.shape[1])])  # PC=principal component
    pca2 = PCA(n_components=pair2.shape[1], random_state=seed)
    pca2_arr = pca2.fit_transform(pair2)
    pair2_pca = pd.DataFrame(pca2_arr,
                             columns=['PC%i' % (ii + 1) for ii in range(pca2_arr.shape[1])])  # PC=principal component

    # K-means plot
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].scatter(pair1_pca['PC1'], pair1_pca['PC2'], c=label1)
    ax[1].scatter(pair2_pca['PC1'], pair2_pca['PC2'], c=label2)
    ax[0].set_title('Pair 1')
    ax[1].set_title('Pair 2')
    plt.show()