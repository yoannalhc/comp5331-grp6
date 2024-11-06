import numpy as np

def fraction_points_changing_cluster(old_clusters, new_clusters):
    changes = np.sum(old_clusters != new_clusters)
    total_points = len(old_clusters)
    return changes / total_points

def solution_cost(points, clusters, medoids):
    max_distance = 0
    for i, point in enumerate(points):
        medoid = medoids[clusters[i]]
        distance = np.linalg.norm(point - points[medoid])
        max_distance = max(max_distance, distance)
    return max_distance

def number_of_clusters(clusters):
    """Count the number of unique clusters formed."""
    return len(np.unique(clusters))