import numpy as np

def distance_matrix(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

def assign_clusters(points, medoids):
    distances = distance_matrix(points)
    cluster_assignment = np.argmin(distances[:, medoids], axis=1)
    return cluster_assignment

def update_medoids(points, clusters, k):
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

def pam(points, k, max_iter=100):
    # Randomly select k initial medoids
    initial_indices = np.random.choice(len(points), k, replace=False)
    medoids = initial_indices.copy()

    for _ in range(max_iter):
        # Step 1: Assign clusters based on current medoids
        clusters = assign_clusters(points, medoids)

        # Step 2: Update medoids
        new_medoids = update_medoids(points, clusters, k)

        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return medoids, clusters