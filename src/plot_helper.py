import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from .evaluation import Metrics
from os.path import join, isdir, isfile
from os import mkdir
from src.datasets import *
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

def plot_cluster_result(pair1, pair2, label1, label2, save_path, ds, algo):

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
    plt.suptitle(f'{ds.name.title()} Clustering Result (k = {ds.k}, {algo})', y=0.98)
    plt.savefig(join(save_path, f'{ds.name}_{ds.k}_{algo}_cluster_result.png'), bbox_inches='tight')
    plt.show()

def plot_bar(results, resilient_k_models, baseline_models, results_path, eval_path, basline_set_random_seed, resilient_set_random_seed):
    metric = Metrics()
    for ds_name, params in results.items():

        cluster_result = {}
        # 1 plot for each k
        flag = False
        if flag and isfile(join(eval_path, f"{ds_name}_baseline_exp_result.csv")):
            ds_result = pd.read_csv(join(eval_path, f"{ds_name}_baseline_exp_result.csv"))
        else:
            ds_result = pd.DataFrame()  # columns=["k", "algo", "fraction_changed", "sol_cost_1", "sol_cost_2", "num_cluster_1", "num_cluster_2"]

        for k in params[0]:
            cluster_result[k] = {}
            # get baseline model here
            for model in baseline_models:
                model_result = {}
                fraction_result_list = []
                sol_cost_result_list = []
                num_cluster_result_list = []
                if basline_set_random_seed:
                    loop_content = params[1]
                else:
                    loop_content = [None] * len(params[1])
                for i, seed in enumerate(loop_content):

                    if seed == None:
                        seed = "None" + "_" + str(i)
                    result_path = f"./{results_path}/baseline/{seed}/{ds_name}"
                    with open(join(result_path, f"{ds_name}_resilient_{k}_{model}_only.pickle"), 'rb') as input_file:
                        center1, cluster1, center2, cluster2 = pickle.load(input_file)
                        cluster1 = [(c[0], np.array([[c[1]]])) for c in cluster1]
                        cluster2 = [(c[0], np.array([[c[1]]])) for c in cluster2]
                        fraction_changed, sol_cost, num_cluster = metric.evaluate(old_points=cluster1,
                                                                                  old_medoids=center1,
                                                                                  new_points=cluster2,
                                                                                  new_medoids=center2, epsilon=0.3)

                        fraction_result_list.append(fraction_changed)
                        sol_cost_result_list.append(sol_cost[1])
                        num_cluster_result_list.append(num_cluster[1])

                fraction_result_list = np.array(fraction_result_list)
                sol_cost_result_list = np.array(sol_cost_result_list)
                num_cluster_result_list = np.array(num_cluster_result_list)

                fraction_mean, fraction_std = np.mean(fraction_result_list), np.std(fraction_result_list)
                sol_cost_mean, sol_cost_std = np.mean(sol_cost_result_list, axis=0), np.std(sol_cost_result_list,
                                                                                            axis=0)
                num_cluster_mean, num_cluster_std = np.mean(num_cluster_result_list, axis=0), np.std(
                    num_cluster_result_list, axis=0)

                ds_result = pd.concat([ds_result, pd.DataFrame.from_records([{
                    "k": k,
                    "algo": f"Baseline {model.title()}",
                    "fraction_changed_mean": fraction_mean,
                    "fraction_changed_std": fraction_std,
                    "sol_cost_1": sol_cost_result_list[0],
                    "sol_cost_2": sol_cost_result_list[1],
                    "sol_cost_mean": sol_cost_mean,
                    "sol_cost_std": sol_cost_std,
                    "num_cluster_1": num_cluster_result_list[0],
                    "num_cluster_2": num_cluster_result_list[1],
                    "num_cluster_mean": num_cluster_mean,
                    "num_cluster_std": num_cluster_std
                }])], ignore_index=True)

                cluster_result[k][f"Baseline {model.title().replace('Hs', 'HS')}"] = {
                    "fraction_of_points_changing_cluster": [fraction_mean, fraction_std],
                    "solution_cost": [sol_cost_mean, sol_cost_std],
                    "number_of_clusters": [num_cluster_mean, num_cluster_std]
                }

                # construct resilient model result below
            # for 1 model result below
            for model in resilient_k_models:
                for a in params[2]:
                    for b in params[3]:
                        fraction_result_list = []
                        sol_cost_result_list = []
                        num_cluster_result_list = []
                        if resilient_set_random_seed:
                            loop_content = params[1]
                        else:
                            loop_content = [None] * len(params[1])
                        for i, seed in enumerate(loop_content):
                            if seed == None:
                                seed = "None" + "_" + str(i)
                            result_path = f"./{results_path}/resilient_k/{seed}/{ds_name}/"
                            with open(join(result_path, f"{ds_name}_resilient_{k}_{model}({a}_{b}).pickle"),
                                      'rb') as input_file:
                                center1, cluster1, center2, cluster2 = pickle.load(input_file)
                            fraction_changed, sol_cost, num_cluster = metric.evaluate(old_points=cluster1,
                                                                                      old_medoids=center1,
                                                                                      new_points=cluster2,
                                                                                      new_medoids=center2, epsilon=0.3)

                            fraction_result_list.append(fraction_changed)

                            sol_cost_result_list.append(sol_cost[1])
                            num_cluster_result_list.append(num_cluster[1])

                        fraction_result_list = np.array(fraction_result_list)
                        sol_cost_result_list = np.array(sol_cost_result_list)
                        num_cluster_result_list = np.array(num_cluster_result_list)

                        fraction_mean, fraction_std = np.mean(fraction_result_list), np.std(fraction_result_list)
                        sol_cost_mean, sol_cost_std = np.mean(sol_cost_result_list, axis=0), np.std(
                            sol_cost_result_list, axis=0)
                        num_cluster_mean, num_cluster_std = np.mean(num_cluster_result_list, axis=0), np.std(
                            num_cluster_result_list, axis=0)

                        cluster_result[k][f'Con{model[0].title()}({a}, {b})'] = {
                            "fraction_of_points_changing_cluster": [fraction_mean, fraction_std],
                            "solution_cost": [sol_cost_mean, sol_cost_std],
                            "number_of_clusters": [num_cluster_mean, num_cluster_std]
                        }
                        ds_result = pd.concat([ds_result, pd.DataFrame.from_records([{
                            "k": k,
                            "algo": f'Con{model[0].title()}({a}, {b})',
                            "fraction_changed_mean": fraction_mean,
                            "fraction_changed_std": fraction_std,
                            "sol_cost_1": sol_cost_result_list[0],
                            "sol_cost_2": sol_cost_result_list[1],
                            "sol_cost_mean": sol_cost_mean,
                            "sol_cost_std": sol_cost_std,
                            "num_cluster_1": num_cluster_result_list[0],
                            "num_cluster_2": num_cluster_result_list[1],
                            "num_cluster_mean": num_cluster_mean,
                            "num_cluster_std": num_cluster_std
                        }])], ignore_index=True)
        flag = True
        for key in ["fraction_of_points_changing_cluster", "solution_cost", "number_of_clusters"]:
            model_results_means = []
            model_results_stds = []
            model_name_list = []
            # get model name
            for k in cluster_result:
                for model in cluster_result[k]:
                    model_name_list.append(model)
                break
            for k in cluster_result:
                model_result_mean = []
                model_result_std = []
                for model in cluster_result[k]:
                    model_result_mean.append(cluster_result[k][model][key][0])
                    model_result_std.append(cluster_result[k][model][key][1])
                model_results_means.append(model_result_mean)
                model_results_stds.append(model_result_std)
            means = np.vstack(model_results_means).T
            stds = np.vstack(model_results_stds).T

            x = np.arange(len(cluster_result))  # the label locations
            width = 0.075  # the width of the bars
            fig = plt.figure()
            # Create bars for each set of data
            for i, (fraction, error) in enumerate(zip(means, stds)):
                plt.bar(x + i * width, fraction, width, label=model_name_list[i], yerr=error, capsize=5)

            # Add some text for labels, title, and custom x-axis tick labels, etc.
            plt.xlabel('k')
            # plt.ylabel(f'{ds_name} {key}'.replace('_', ' ').title())
            plt.title(f'{ds_name} {key}'.replace('_', ' ').title())
            plt.xticks(x + width * (len(model_name_list) - 1) / 2, cluster_result.keys())
            lgd = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            if key == "fraction_of_points_changing_cluster":
                ax = plt.gca()
                ax.set_ylim([0.0, 1.0])
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

            # Show the plot
            if not isdir(join(results_path, "plots")):
                mkdir(join(results_path, "plots"))
            plot_dir = f"./{results_path}/plots/{ds_name}"
            if not isdir(plot_dir):
                mkdir(plot_dir)

            plt.savefig(join(plot_dir, f"{ds_name}_{key}.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')

        ds_result.to_csv(join(eval_path, f"{ds_name}_exp_result.csv"), index=False)