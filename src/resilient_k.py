import numpy as np
import copy
import random
import math

# helper function
def distance(point1, point2):
    # This function is to calculate the distance between 2 points
    distance_x_y = np.linalg.norm(np.array(point1) - np.array(point2))
    return distance_x_y

# helper class
# Modified from https://dev.to/theramoliya/python-kruskals-algorithm-for-minimum-spanning-trees-2bmb
class Graph:
    def __init__(self, vertices, graph=None):
        self.V = vertices
        if graph is None:
            self.graph = []
        else:
            self.graph = copy.deepcopy(graph)

    def copy(self):
        return Graph(self.V, self.graph)

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def remove_edge(self, u, v, w):
        self.graph.remove([u, v, w])

    def kruskal_mst(self):
        def find(parent, i):
            if parent[i] == i:
                return i
            return find(parent, parent[i])

        def union(parent, rank, x, y):
            x_root = find(parent, x)
            y_root = find(parent, y)

            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            elif rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[y_root] = x_root
                rank[x_root] += 1

        result = []
        i = 0
        e = 0

        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = [i for i in range(self.V)]
        rank = [0] * self.V

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = find(parent, u)
            y = find(parent, v)

            if x != y:
                e += 1
                result.append([u, v, w])
                union(parent, rank, x, y)

        return result


class Gonz_Approx_Algo:
    def __init__(self, dataset, k, seed=5331):
        self.dataset = dataset
        self.k = k
        self.seed = seed

    # Define KCluster
    class Cluster:
        def __init__(self):
            self.elements = []  # Initially, cluster has no points inside
            self.head = None

    def clustering(self):
        def initialize_clusters(dataset, seed=None):
            # Since we need to initialize the first cluster, there must be someone who does that first
            cluster = Gonz_Approx_Algo.Cluster()  # call class Cluster
            cluster.elements = dataset.tolist()  # All data now become the point of cluster 1
            if seed is not None:
                random.seed(seed)
            cluster.head = random.choice(cluster.elements)
            return [cluster]

        def expand_clusters(clusters, j):
            # At this function, we will perform expansion

            # We will find the point with maximal distance to the head
            max_distance = -1
            v_i = None

            for i in range(j - 1):
                current_cluster = clusters[i]

                for point in current_cluster.elements:
                    dist = distance(point, current_cluster.head)
                    if dist > max_distance:
                        max_distance = dist
                        v_i = point

            # Create new cluster B_(j + 1)
            new_cluster = Gonz_Approx_Algo.Cluster()
            new_cluster.head = v_i
            new_cluster.elements = []

            # Move elements to the new cluster
            for i in range(j - 1):
                current_cluster = clusters[i]
                # print("head ", i+1, current_cluster.head)
                for point in current_cluster.elements:
                    # print("distance v_1", distance(point, v_i))
                    # print("distance head", distance(point, current_cluster.head))
                    if distance(point, v_i) <= distance(point, current_cluster.head):
                        # print("point ", point)
                        new_cluster.elements.append(point)

                # Delete the elements that was appended to new cluster
                current_cluster.elements = [element for element in current_cluster.elements if
                                            element not in new_cluster.elements]
                # print("Cluster ", i + 1, " : ", current_cluster.elements)

            # Add this new cluster to a list of cluster
            clusters.append(new_cluster)

            return clusters

        def get_heads(clusters):
            # Give me the list of current clusters head
            heads = []
            for cluster in clusters:
                heads.append(cluster.head)

            return heads

        clusters = initialize_clusters(self.dataset, self.seed)
        for self.k in range(2,
                            self.k + 1):  # note that it should be range(2, k+1), we start from 2 because we already initialize a cluster
            clusters = expand_clusters(clusters, self.k)

        # Get the heads of the clusters
        heads = get_heads(clusters)

        return np.asarray(heads)


class resilient_k_center():
    def __init__(self, dataset, k, epsilon, lamb=0.1, alpha=1.0, beta=1.0, algorithm="gonz", seed=5331):
        if alpha != 0.5 and alpha != 1.0:
            raise ValueError("alpha must be 0.5 or 1.0")
        if beta != 0.5 and beta != 1.0:
            raise ValueError("beta must be 0.5 or 1.0")
        if algorithm != "gonz" and algorithm != "carv":
            raise ValueError("algorithm must be gonz or carv")
        self.dataset = dataset
        self.k = k
        self.epsilon = epsilon
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.random_centers = int(self.alpha * 2 * self.k * np.log(1 / self.epsilon))
        self.algorithm = algorithm
        self.algorithm_centers = int(self.beta * self.k)
        self.seed = seed

    def resilient_k_center(self):
        # randomly assign centers (line 1)
        centers = self.dataset[np.random.choice(self.dataset.shape[0],
                                                self.random_centers,
                                                replace=False)]
        #print("Centers: \n:", centers)

        # construct edges and weights (line 2-4)
        E = []
        w = {}
        for index_p, p in enumerate(self.dataset):
            for index_q, q in enumerate(self.dataset):
                if index_p < index_q:
                    if (p.tolist() in centers.tolist()) or (q.tolist() in centers.tolist()):
                        E.append((index_p, index_q))

                        # Integrated line 1-7 of algorithm Resilient-MST in the weight update below
                        if (p.tolist() in centers.tolist()) and (q.tolist() in centers.tolist()):
                            w[(index_p, index_q)] = 0
                        else:
                            alpha = np.random.rand()
                            i = math.ceil(alpha + math.log(distance(p, q), self.lamb))
                            w[(index_p, index_q)] = self.lamb ** (i - alpha)
        #print(E)
        #print(w)

        # construct weighted graph (line 5)
        g = Graph(len(self.dataset))
        for edge in E:
            index_p, index_q = edge
            g.add_edge(index_p, index_q, w[(index_p, index_q)])
            g.add_edge(index_q, index_p, w[(index_p, index_q)])
        #print("weighted graph: \n", g.graph)

        # construct MST (line 6)
        T = g.kruskal_mst()
        #print("resilient MST: \n", T)

        # assign clusters (line 7-8)
        cluster = []
        for index_p, p in enumerate(self.dataset):
            if p.tolist() in centers.tolist():
                cluster.append((p, p))
            else:
                for edge in T:
                    index_u, index_v, _ = edge
                    if (index_p == index_u) and (self.dataset[index_v].tolist() in centers.tolist()):
                        cluster.append((p, self.dataset[index_v]))
                    elif (index_p == index_v) and (self.dataset[index_u].tolist() in centers.tolist()):
                        cluster.append((p, self.dataset[index_u]))

        #print("Initial Cluster: \n", cluster)

        # find non-center vertices which incident to the heaviest edges (line 9)
        n = len(self.dataset)
        heaviest_edges = sorted(T, key=lambda item: item[2], reverse=True)[:int(self.epsilon * n)]
        #print("Heaviest Edges: \n", heaviest_edges)
        L = set()
        for edge in heaviest_edges:
            index_p, index_q, _ = edge
            if (self.dataset[index_p].tolist() not in centers.tolist()):
                L.add(index_p)
            if (self.dataset[index_q].tolist() not in centers.tolist()):
                L.add(index_q)
        #print("L: \n", L)

        # centres selected by Algorithm Approx [27] (line 10)
        centers_approx = None
        if self.algorithm == "gonz":
            approx_algo = Gonz_Approx_Algo(self.dataset, self.algorithm_centers, self.seed)
            centers_approx = approx_algo.clustering()
        elif self.algorithm == "carv":
            pass 
        else:
            raise ValueError("Algorithm not supported")
        #print("Centers selected by Approx: \n", centers_approx)

        # Update the center as the closest center of C' to each point in L (line 11)
        for index_p in L:
            min_dist = float('inf')
            closest_center = None
            for c in centers_approx:
                dist = distance(self.dataset[index_p], c)
                if dist < min_dist:
                    min_dist = dist
                    closest_center = c
            if (closest_center is not None):
                for index_c, cluster_pair in enumerate(cluster):
                    p, _ = cluster_pair
                    if np.array_equal(p, self.dataset[index_p]):
                        cluster[index_c] = (p, closest_center)
                        break

        # get final center list
        centers_final = np.unique([c[1] for c in cluster], axis=0)
        #print("cluster center:", centers_final)
        # label each data belongs to which center in centers_final with its idx
        labels = []
        for data, center in cluster:
            idx = np.where(np.all(centers_final == center, axis=1))
            labels.append((data, idx))
        #print("Final Cluster: \n", labels)
        return centers_final, labels