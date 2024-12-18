import numpy as np
import copy
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
    def __init__(self, dataset, k, seed=None):
        self.dataset = dataset
        self.k = k
        self.seed = seed

    # Define KCluster
    class Cluster:
        def __init__(self):
            self.elements = []  # Initially, cluster has no points inside
            self.head = None

    def clustering(self):
        def initialize_clusters(dataset):
            # Since we need to initialize the first cluster, there must be someone who does that first
            cluster = Gonz_Approx_Algo.Cluster()  # call class Cluster
            cluster.elements = dataset.tolist()  # All data now become the point of cluster 1
            if self.seed is not None:
                np.random.seed(self.seed)
            
            head_idx = np.random.choice(len(cluster.elements), replace=False)
            
            cluster.head = cluster.elements[head_idx]  # Randomly choose the head of the cluster
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

        clusters = initialize_clusters(self.dataset)
        for self.k in range(2,
                            self.k + 1):  # note that it should be range(2, k+1), we start from 2 because we already initialize a cluster
            clusters = expand_clusters(clusters, self.k)

        # Get the heads of the clusters
        heads = get_heads(clusters)

        return np.asarray(heads)


class CarvingAlgorithm:
    def __init__(self, points, seed=None):
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

    def carve(self, R):
        """Perform the carving algorithm with the given radius R and number of centers k."""
        centers = []
        uncovered_indices = set(range(len(self.points)))  # Set of indices of uncovered points
        if self.seed is not None:
            np.random.seed(self.seed)

        # while uncovered_indices and len(centers) < k:
        while uncovered_indices:
            # Randomly select an uncovered point
            uncover_point_index = np.random.choice(len(list(uncovered_indices)), replace=False)
            idx = list(uncovered_indices)[uncover_point_index]
            center = self.points[idx]
            centers.append(center)

            # Mark all points within distance R from the new center as covered
            to_remove = {i for i in uncovered_indices if self.distance(center, self.points[i]) <= R}
            uncovered_indices.difference_update(to_remove)

        return centers

    def find_minimum_R(self, k):
        best_R = None
        R_start = 0  # every point is a center
        R_end = self.find_farthest_point_distance()  # one point is centre and all other points are within R distance
        R_mid = (R_start + R_end) // 2
        while R_end != R_start + 1:
            centers = self.carve(R_mid)
            # print("R_mid: ", R_mid, "Centers: ", len(centers), "k: ", k, "best_R: ", best_R)
            if len(centers) <= k:
                best_R = R_mid
                R_end = R_mid
            else:
                R_start = R_mid
            R_mid = (R_start + R_end) // 2
            # print("R_start: ", R_start, "R_end: ", R_end, "R_mid: ", R_mid)
        # print("Best R: ", best_R)
        return best_R


class resilient_k_center():
    def __init__(self, dataset, k, epsilon, lamb=0.1, alpha=1.0, beta=1.0, algorithm="gonz", seed=None):
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
        self.random_centers = int(self.alpha * self.k)
        self.algorithm = algorithm
        self.algorithm_centers = int(self.beta * self.k)
        self.seed = seed
    def resilient_k_center(self):
        # randomly assign centers (line 1)
        if self.seed is not None:
            np.random.seed(self.seed)
        centers = self.dataset[np.random.choice(self.dataset.shape[0],
                                                self.random_centers,
                                                replace=False)]

        
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
        # cluster shape = n (coordinate, cluster coordinate)
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
        assert len(cluster) == len(self.dataset)
        # at this moment, len(cluster) != n
        
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
            approx_algo = CarvingAlgorithm(self.dataset,seed=self.seed)
            best_r = approx_algo.find_minimum_R(self.algorithm_centers)
            centers_approx = approx_algo.carve(best_r)
        else:
            raise ValueError("Algorithm not supported")
        #print("Centers selected by Approx: \n", centers_approx)

        # Update the center as the closest center of C' to each point in L (line 11)
        for index_p in L: # L = P\C 
            min_dist = float('inf')
            closest_center = None
            for c in centers_approx: # C'
                dist = distance(self.dataset[index_p], c)
                if dist < min_dist:
                    min_dist = dist
                    closest_center = c
                    
            # loop through all point in cluster
            assert len(cluster) == n
            for index_c, cluster_pair in enumerate(cluster): # len(cluster) != n
                p, _ = cluster_pair
                # only doing reassign?
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