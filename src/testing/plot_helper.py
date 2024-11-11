import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from os.path import join
import numpy as np
import matplotlib.ticker as mtick

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
        plt.tight_layout()

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
        plt.tight_layout()

    fig.subplots_adjust(top=0.8)

    plt.suptitle(f'{ds_name} dataset', y=0.98)
    plt.savefig(join(save_path, f'{ds_name}_visual.png'), bbox_inches='tight')
    plt.show()

def plot_cluster_result(pair1, pair2, label1, label2, save_path, ds, algo, alpha, beta):

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
        plt.tight_layout()
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
        plt.tight_layout()

    fig.subplots_adjust(top=0.8)
    plt.suptitle(f'{ds.name.title()} Clustering Result (k = {ds.k}, {algo.title()}({alpha}, {beta}))', y=0.98)
    plt.savefig(join(save_path, f'{ds.name}_{ds.k}_{algo}({alpha}_{beta})_cluster_result.png'), bbox_inches='tight')
    #plt.show()
    # if show:
    #     plt.show()
    # else:
    #     plt.clf()
    plt.close()

def plot_result_bar(resilient_df, baseline_df, ds_name, lamb, save_path):
    plot_df = pd.concat([baseline_df, resilient_df], ignore_index=True)
    k_df = plot_df.groupby('k')

    print("keys:", k_df.groups.keys())
    keys_list = list(k_df.groups.keys())
    grp = k_df.get_group(keys_list[0])
    algo_list = grp['algo'].to_list()
    num_algo = len(algo_list)

    # Determine bar widths
    width = 0.1
    margin = 1.5 # space between k=10, k=20
    X_axis = np.asarray([i * margin for i in range(len(keys_list))])
    #print("num_algo", num_algo)
    #print("algo_list", algo_list)
    fig, ax = plt.subplots(figsize=(6,5))
    bars = []
    for i, (k, grp) in enumerate(k_df):
        grp.set_index('algo', inplace=True)
        #print("index:", grp.index)
        for j, algo in enumerate(grp.index):
            #shift = [width * j for j in range(len(grp['algo'].index))]
            #x_positions = k + (width_bar * i) - width_cluster / 2
            #bars.append(ax.bar(x_positions, grp['fraction_changed'], width_bar, align='edge', label=algo))
            #print("algo:", grp['fraction_changed'].iloc[j])
            bars.append(ax.bar(i * margin + width * j, grp['fraction_changed'][j], width, label=algo))

        #plt.bar(X_axis + shift, k_20['fraction_changed'], width, label=k_20['algo'])
    plt.xticks(X_axis + width * (num_algo-1)/ 2, ['10', '20'])
    plt.xlabel("k")
    plt.ylabel("Fraction of Points Changing Clusters")
    plt.title(f"{ds_name} Fraction of Points Changing Clusters ("+ r"$\lambda = $"+ f"{lamb})")

    ax = fig.gca()
    for i, p in enumerate(bars):  # this is the loop to change Labels and colors
        if p.get_label() in algo_list[:i]:  # check for Name already exists
            idx = algo_list.index(p.get_label())  # find ist index
            p.patches[0].set_facecolor(bars[idx].patches[0].get_facecolor())  # set color
            p.set_label('_' + p.get_label())

    # Shrink current axis by 20%
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.subplots_adjust(right=0.7)
    #plt.tight_layout(rect=[0, 0, 0.75, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    #plt.legend( loc='upper right')
    plt.savefig(join(save_path, f'{ds_name}_frac_change_result.png'), bbox_inches='tight')
    plt.show()