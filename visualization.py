import matplotlib.pyplot as plt
from evaluation import Metrics
'''
Things to plot:
1. Data points and Centers
2. Plot for each metrics
'''
def plot_clusters(X, y, centers, title, dimension_reduction_method = 'pca'):
    # if X is multidimensional perform pca or tsne
    if X.shape[1] > 2:
        if dimension_reduction_method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
        elif dimension_reduction_method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            X = tsne.fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    plt.show()

def plot_fraction_points_changing_cluster(fpccs, algorithms, dataset):
    # plot barchart fraction of points changing cluster for each algorithm performed on same dataset
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = range(len(fpccs))
    for i in range(len(fpccs)):
        ax.bar(index[i], fpccs[i], bar_width, label=algorithms[i])
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Fraction of points changing cluster')
    ax.set_title('Fraction of points changing cluster for different algorithms on dataset: {}'.format(dataset))
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    ax.legend()
    plt.show()
    

def plot_number_of_clusters(number_of_clusters, algorithms, dataset):
    # plot bar chart for number of changing clusters for each algorithm
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = range(len(number_of_clusters))
    for i in range(len(number_of_clusters)):
        ax.bar(index[i], number_of_clusters[i], bar_width, label=algorithms[i])
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Number of clusters')
    ax.set_title('Number of clusters for different algorithms on dataset: {}'.format(dataset))
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    ax.legend()
    plt.show()

def plot_solution_cost(solution_costs, algorithms, dataset):
    # plot solution cost for each algorithm
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = range(len(solution_costs))
    for i in range(len(solution_costs)):
        ax.bar(index[i], solution_costs[i], bar_width, label=algorithms[i])
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Solution cost')
    ax.set_title('Solution cost for different algorithms on dataset: {}'.format(dataset))
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    metric = Metrics()
    metric.fraction_points_changing_cluster()
    metric.number_of_clusters()
    metric.solution_cost()