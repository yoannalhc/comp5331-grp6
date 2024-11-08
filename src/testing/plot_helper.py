import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from os.path import join
import numpy as np

def plot_data(pair1, pair2, save_path, ds_name):

    # Compute PCA of data set
    if pair1.shape[1] > 3:
        seed = 0
        pca1 = PCA(n_components=pair1.shape[1], random_state=seed)
        pca1_arr = pca1.fit_transform(pair1)
        pair1_pca = pd.DataFrame(pca1_arr,
                             columns=['PC%i' % (ii + 1) for ii in range(pca1_arr.shape[1])])  # PC=principal component
        pca2 = PCA(n_components=pair2.shape[1], random_state=seed)
        pca2_arr = pca2.fit_transform(pair2)
        pair2_pca = pd.DataFrame(pca2_arr,
                                 columns=['PC%i' % (ii + 1) for ii in range(pca2_arr.shape[1])])  # PC=principal component

        # 3D PCA K-means plot
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(pair1_pca['PC1'], pair1_pca['PC2'], pair1_pca['PC3'])
        ax.set_title('Pair 1')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(pair2_pca['PC1'], pair2_pca['PC2'], pair2_pca['PC3'])
        ax.set_title('Pair 2')

        # PCA K-means plot
        # fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        # ax[0].scatter(pair1_pca['PC1'], pair1_pca['PC2'], c=label1)
        # ax[1].scatter(pair2_pca['PC1'], pair2_pca['PC2'], c=label2)
        # ax[0].set_title('Pair 1')
        # ax[1].set_title('Pair 2')
    elif pair1.shape[1] == 3:
        pair1 = pair1.T
        pair2 = pair2.T
        # 3D K-means plot
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(pair1[0], pair1[1], pair1[2])
        ax.set_title('Pair 1')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(pair2[0], pair2[1], pair2[2])
        ax.set_title('Pair 2')
    elif pair1.shape[1] == 2:
        pair1 = pair1.T
        pair2 = pair2.T
        # 2D K-means plot
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].scatter(pair1[0], pair1[1])
        ax[1].scatter(pair2[0], pair2[1])
        ax[0].set_title('Pair 1')
        ax[1].set_title('Pair 2')
    else:
        pair1 = pair1.T
        pair2 = pair2.T
        # 1D K-means plot
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ar = np.arange(pair1.shape[1])
        ax[0].scatter(pair1[0], np.zeros_like(ar))
        ax[1].scatter(pair2[0], np.zeros_like(ar))
        ax[0].set_title('Pair 1')
        ax[1].set_title('Pair 2')

    plt.suptitle(f'{ds_name} dataset')
    plt.savefig(join(save_path, f'{ds_name}_visual.png'), bbox_inches='tight')
    plt.show()

def plot_cluster_result(pair1, pair2, label1, label2, save_path, ds):

    # Compute PCA of data set
    if pair1.shape[1] > 3:
        seed = 0
        pca1 = PCA(n_components=pair1.shape[1], random_state=seed)
        pca1_arr = pca1.fit_transform(pair1)
        pair1_pca = pd.DataFrame(pca1_arr,
                             columns=['PC%i' % (ii + 1) for ii in range(pca1_arr.shape[1])])  # PC=principal component
        pca2 = PCA(n_components=pair2.shape[1], random_state=seed)
        pca2_arr = pca2.fit_transform(pair2)
        pair2_pca = pd.DataFrame(pca2_arr,
                                 columns=['PC%i' % (ii + 1) for ii in range(pca2_arr.shape[1])])  # PC=principal component

        # 3D PCA K-means plot
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(pair1_pca['PC1'], pair1_pca['PC2'], pair1_pca['PC3'], c=label1)
        ax.set_title('Pair 1')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(pair2_pca['PC1'], pair2_pca['PC2'], pair2_pca['PC3'], c=label2)
        ax.set_title('Pair 2')

        # PCA K-means plot
        # fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        # ax[0].scatter(pair1_pca['PC1'], pair1_pca['PC2'], c=label1)
        # ax[1].scatter(pair2_pca['PC1'], pair2_pca['PC2'], c=label2)
        # ax[0].set_title('Pair 1')
        # ax[1].set_title('Pair 2')
    elif pair1.shape[1] == 3:
        pair1 = pair1.T
        pair2 = pair2.T
        # 3D K-means plot
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(pair1[0], pair1[1], pair1[2], c=label1)
        ax.set_title('Pair 1')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(pair2[0], pair2[1], pair2[2], c=label2)
        ax.set_title('Pair 2')
    elif pair1.shape[1] == 2:
        pair1 = pair1.T
        pair2 = pair2.T
        # 2D K-means plot
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].scatter(pair1[0], pair1[1], c=label1)
        ax[1].scatter(pair2[0], pair2[1], c=label2)
        ax[0].set_title('Pair 1')
        ax[1].set_title('Pair 2')
    else:
        pair1 = pair1.T
        pair2 = pair2.T
        # 1D K-means plot
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ar = np.arange(pair1.shape[1])
        ax[0].scatter(pair1[0], np.zeros_like(ar), c=label1)
        ax[1].scatter(pair2[0], np.zeros_like(ar), c=label2)
        ax[0].set_title('Pair 1')
        ax[1].set_title('Pair 2')

    plt.suptitle(f'{ds.name.title()} Clustering Result (k = {ds.k})')
    plt.savefig(join(save_path, f'{ds.name}_{ds.k}_cluster_result.png'), bbox_inches='tight')
    plt.show()