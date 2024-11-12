'''
# Types of clustering objectives
Global objective: 

K-means:
Object: minimize the sum of squared distance from each item to its nearest averaged center.
K-centers:
Object: minimize the maximum distance from each item to its nearest cluster centers
k-medians:
Object: minimize the sum of distance from each item to its nearest median. 
k-medoids:
Object: minimize the sum of squared distance from each item to its nearest medoids.
'''

import numpy as np
from scipy.spatial import distance
import math
import copy
import random 
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices  
        self.edges = [] # (index_u, index_v, weight)
        self.adj_list = dict() # {index_u: [(index_v, weight)]}

    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))
    
    def update_adj_list(self):
        adj_list = {i: [] for i in range(self.V)}

        print("Edges: ", self.edges)
        for index_u, index_v, weight in self.edges:
            adj_list[index_u].append((index_v, weight))
            if (index_u != index_v):
                adj_list[index_v].append((index_u, weight))
        self.adj_list = adj_list
        print("Adjacency list: \n", self.adj_list)

    def sort(self):
        self.edges.sort(key=lambda x: x[2])

############ k center algorithm ################

class CarvingAlgorithm:
    def __init__(self, points, seed=5331):
        self.points = np.array(points)
        self.seed = seed

    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)

    def find_farthest_point_distance(self):
        """Find the maximum distance between any two points in the dataset."""
        max_distance = 0
        for i, point1 in enumerate(self.points):
            for j in range(i + 1, len(self.points)):
                point2 = self.points[j]
                distance = self.distance(point1, point2)
                max_distance = max(max_distance, distance)
        # print("Max distance: ", max_distance)
        return max_distance

    def carve(self, R, k, seed=5331, is_clustering=True):
        """Perform the carving algorithm with the given radius R and number of centers k."""
        centers = []
        uncovered_indices = set(range(len(self.points)))  # Set of indices of uncovered points
        if (seed is not None):
            random.seed(seed)

        # while uncovered_indices and len(centers) < k:
        while uncovered_indices:
            # Randomly select an uncovered point
            idx = random.choice(list(uncovered_indices))
            center = self.points[idx]
            centers.append(center)

            # Mark all points within distance R from the new center as covered
            to_remove = {i for i in uncovered_indices if self.distance(center, self.points[i]) <= R}
            uncovered_indices.difference_update(to_remove)
        # add label for points
        if is_clustering:
            labels = self.assign_labels(centers)
            return centers, labels
        labels = self.assign_labels(centers)
        return centers, labels
    def assign_labels(self, clusters):
        labels = []
        for point in self.points:
            # Find the nearest cluster for each point
            nearest_center_index = np.argmin([distance.euclidean(point, center) for center in clusters])
            labels.append((point, nearest_center_index))  # (data point, index of nearest center)
        return labels
    def find_minimum_R(self, k):
        best_R = None
        R_start = 0  # every point is a center
        R_end = self.find_farthest_point_distance()  # one point is centre and all other points are within R distance
        R_mid = (R_start + R_end) // 2
        while R_end != R_start + 1:
            centers = self.carve(R_mid, k, is_clustering=False)
            # print("R_mid: ", R_mid, "Centers: ", len(centers), "k: ", k, "best_R: ", best_R)
            if len(centers) <= k:
                best_R = R_mid
                R_end = R_mid
            else:
                R_start = R_mid
            R_mid = (R_start + R_end) // 2
            # print("R_start: ", R_start, "R_end: ", R_end, "R_mid: ", R_mid)
        print("Best R: ", best_R)
        return best_R
    
class GonzalezAlgorithm:
    # Modified from https://github.com/TSunny007/Clustering/blob/master/notebooks/Gonzalez.ipynb
    def __init__(self, points, cluster_num):
        self.points = np.array(points)
        self.cluster_num = cluster_num
    
    def max_dist(self, data, clusters):
        distances = np.zeros(len(data))
        for cluster_id, cluster in enumerate(clusters):
            for point_id, point in enumerate(data):
                if distance.euclidean(point,cluster) == 0.0:
                    distances[point_id] = -math.inf # this point is already a cluster 
                if not math.isinf(distances[point_id]):
                    distances[point_id] = distances[point_id] + distance.euclidean(point,cluster) 
        return data[np.argmax(distances)]

    def norm_dist(self, data, clusters):
        distances = np.zeros(len(data)) 
        for point_id, point in enumerate(data):
            for cluster_id, cluster in enumerate(clusters):
                if distance.euclidean(point,cluster) == 0.0:
                    distances[point_id] = -math.inf 
                if not math.isinf(distances[point_id]):
                    distances[point_id] = distances[point_id] + math.pow(distance.euclidean(point,cluster),2) 
        for distance_id, current_distance in enumerate(distances):
            if not math.isinf(current_distance): 
                distances[distance_id] = math.sqrt(current_distance/len(data))
        return data[np.argmax(distances)]

    def gonzalez(self, method = 'max'):
        clusters = []
        # random choose first point
        clusters.append(random.choice(self.points)) # assign the first point to the first cluster
        while len(clusters) < self.cluster_num:
            if method == 'max':
                clusters.append(self.max_dist(self.points, clusters)) 
            if method == 'norm':
                clusters.append(self.norm_dist(self.points, clusters)) 
            # we add the furthest point from ALL current clusters
        labels = self.assign_labels(clusters)
        return clusters, labels

    def assign_labels(self, clusters):
        labels = []
        for point in self.points:
            # Find the nearest cluster for each point
            nearest_center_index = np.argmin([distance.euclidean(point, center) for center in clusters])
            labels.append((point, nearest_center_index))  # (data point, index of nearest center)
        return labels
    
class HSAlgorithm:
    def __init__(self, points, cluster_num):
        self.points = np.array(points)
        self.cluster_num = cluster_num

    
    def hs_algorithm(self, points, k):
        n = len(points)
        centers = []
        labels = []  
        # set random seed = 5331
        np.random.seed(5331)
        initial_center_index = np.random.choice(n)
        centers.append(points[initial_center_index])

        for center_index in range(1, k):
            max_distance = -1
            next_center_index = None

            for i in range(n):
                # Find the distance to the nearest center
                nearest_distance = min(distance.euclidean(points[i], centers[j]) for j in range(center_index))
                
                # Find the point with the maximum nearest distance
                if nearest_distance > max_distance:
                    max_distance = nearest_distance
                    next_center_index = i

            # Add the farthest point as the next center
            centers.append(points[next_center_index])

        for i in range(n):
            nearest_center_index = np.argmin([distance.euclidean(points[i], center) for center in centers])
            labels.append((points[i], nearest_center_index))

        return centers, labels

############ k medoid algorithm ################

class PAMAlgorithm:
    def __init__(self, points, cluster_num):
        self.points = np.array(points)
        self.cluster_num = cluster_num
        
    def distance_matrix(self, points):
        return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    def assign_clusters(self, points, medoids):
        distances = self.distance_matrix(points)
        cluster_assignment = np.argmin(distances[:, medoids], axis=1)
        return cluster_assignment

    def update_medoids(self, points, clusters, k):
        new_medoids = []
        for i in range(k):
            # Get points in the current cluster
            cluster_points = points[clusters == i]
            if len(cluster_points) == 0:
                continue

            # Calculate the cost for each point in the cluster as a potential medoid
            costs = []
            for point in cluster_points:
                cost = np.sum(np.linalg.norm(cluster_points - point, axis=1))
                costs.append(cost)

            # Find the point with the minimum cost
            new_medoid = cluster_points[np.argmin(costs)]
            new_medoids.append(np.where((points == new_medoid).all(axis=1))[0][0])

        return np.array(new_medoids)

    def pam(self, max_iter=100):
        # Randomly select k initial medoids
        initial_indices = np.random.choice(len(self.points), self.cluster_num, replace=False)
        medoids = initial_indices.copy()

        for _ in range(max_iter):
            # Step 1: Assign clusters based on current medoids
            clusters = self.assign_clusters(self.points, medoids)

            # Step 2: Update medoids
            new_medoids = self.update_medoids(self.points, clusters, self.cluster_num)

            # Check for convergence
            if np.array_equal(medoids, new_medoids):
                break

            medoids = new_medoids

        return medoids, clusters
    
class SpectralClustering:
    def __init__(self, points, cluster_num):
        self.points = np.array(points)
        self.cluster_num = cluster_num

    def distance_matrix(self, points):
        return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    def spectral_clustering(self):
        distances = self.distance_matrix(self.points)
        sigma = np.median(distances)
        similarity_matrix = np.exp(-distances ** 2 / (2 * sigma ** 2))

        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
        laplacian_matrix = degree_matrix - similarity_matrix 
        eigenvalues, eigenvectors = eigh(laplacian_matrix, eigvals=(1, self.cluster_num))

        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
        cluster_assignment = kmeans.fit_predict(eigenvectors)

        return cluster_assignment
    
class KDBSCAN:
    # ROCK: A Robust Clustering Algorithm for Categorical Attributes
    def __init__(self, points, cluster_num, eps, min_samples, k):
        self.points = np.array(points)
        self.cluster_num = cluster_num

    def fit(self, X):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(X)
        unique_labels = set(self.labels_)
        clusters = [X[self.labels_ == label] for label in unique_labels if label != -1]  # Exclude noise
        
        clusters.sort(key=len, reverse=True)  
        selected_clusters = clusters[:self.k]
        
        final_labels = np.full(X.shape[0], -1)  
        for i, cluster in enumerate(selected_clusters):
            final_labels[self.labels_ == list(unique_labels)[i + 1]] = i  # Assign cluster labels

        # Assign non-selected points to the nearest selected cluster
        for i, cluster in enumerate(clusters[self.k:]):
            if len(cluster) > 0:
                distances = pairwise_distances_argmin_min(cluster, np.vstack(selected_clusters))
                nearest_cluster_index = distances[0]  
                final_labels[self.labels_ == list(unique_labels)[i + self.k + 1]] = nearest_cluster_index

        return final_labels

class Clustering():
    def __init__(self, n_clusters, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    # K-means
    
    # K-centers
    def gonzalez(self, data, method = 'max'):
        gonzalez_fn = GonzalezAlgorithm(data, self.n_clusters)
        centers = gonzalez_fn.gonzalez(method)
        return centers

    def carve(self, R, data):
        carve_fn = CarvingAlgorithm(data)
        best_r = carve_fn.find_minimum_R(0, R, self.n_clusters)
        centers = carve_fn.carve(best_r, self.n_clusters)
        return centers
    # K-medians
    
    
    # K-medoids
    def pam(self, data):
        pam_fn = PAMAlgorithm(data, self.n_clusters)
        medoids, clusters = pam_fn.pam(self.max_iter)
        return medoids, clusters

    # Others
    def spectral_clustering(self, data):
        spectral_fn = SpectralClustering(data, self.n_clusters)
        cluster_assignment = spectral_fn.spectral_clustering()
        return cluster_assignment

    # ROCK: A Robust Clustering Algorithm for Categorical Attributes

    '''
    Baselines from paper to be implemented:
    Fast Distributed k-Center Clustering with Outliers on Massive Data
    Solving k-center Clustering (with Outliers) in MapReduce and Streaming, almost as Accurately as Sequentially
    Fast Distributed k-Center Clustering with Outliers on Massive Data
    Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction
    A Composable Coreset for k-Center in Doubling Metrics
    Randomized Greedy Algorithms and Composable Coreset for k-Center Clustering with Outliers
    k-Center Clustering with Outliers in Sliding Windows
    Dimensionality-adaptive k-center in sliding windows
    '''
    
 
if __name__ == "__main__":
    # read data
    # csv_file = "Clustering-Datasets/01. UCI/ecoli.csv"
    # data = np.genfromtxt(csv_file, delimiter=',')
    
    # test
    # cluster_algos = Clustering()

    # Carving
    import matplotlib.pyplot as plt
    import numpy as np

    # Example data preparation
    models = ['Gonz', 'Carv', 'ConG(0.5,0.5)', 'ConG(1.0,0.5)', 'ConG(1.0,1.0)', 'ConG(1.5,1.0)']
    k_values = ['k = 10', 'k = 20']

    # Example data for different seeds (replace with actual values)
    seed_results = {
        'Gonz': {
            'k = 10': [0.1, 0.11, 0.09],
            'k = 20': [0.2, 0.21, 0.19]
        },
        'Carv': {
            'k = 10': [0.2, 0.22, 0.19],
            'k = 20': [0.3, 0.31, 0.29]
        },
        'ConG(0.5,0.5)': {
            'k = 10': [0.15, 0.14, 0.16],
            'k = 20': [0.25, 0.26, 0.24]
        },
        'ConG(1.0,0.5)': {
            'k = 10': [0.05, 0.06, 0.04],
            'k = 20': [0.1, 0.11, 0.09]
        },
        'ConG(1.0,1.0)': {
            'k = 10': [0.12, 0.13, 0.11],
            'k = 20': [0.18, 0.19, 0.17]
        },
        'ConG(1.5,1.0)': {
            'k = 10': [0.09, 0.1, 0.08],
            'k = 20': [0.14, 0.15, 0.13]
        },
    }

    # Calculate mean and std deviation for error bars
    means_k10 = []
    stds_k10 = []
    means_k20 = []
    stds_k20 = []

    for model in models:
        means_k10.append(np.mean(seed_results[model]['k = 10']))
        stds_k10.append(np.std(seed_results[model]['k = 10']))
        means_k20.append(np.mean(seed_results[model]['k = 20']))
        stds_k20.append(np.std(seed_results[model]['k = 20']))

    # Bar width and positions
    bar_width = 0.35
    index = np.arange(len(models))

    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting bars for k = 10
    bars1 = ax.bar(index - bar_width/2, means_k10, bar_width,
                    label='k = 10', yerr=stds_k10, capsize=5, color='lightcoral', alpha=0.6)

    # Plotting bars for k = 20
    bars2 = ax.bar(index + bar_width/2, means_k20, bar_width,
                    label='k = 20', yerr=stds_k20, capsize=5, color='skyblue', alpha=0.6)

    # Customizing the plot
    ax.set_title('Fraction of Points Changing Cluster')
    ax.set_xlabel('Models')
    ax.set_ylabel('Fraction')
    ax.set_xticks(index)
    ax.set_xticklabels(models)
    ax.legend()

    # Displaying the plot
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()