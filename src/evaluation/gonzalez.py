from scipy.spatial import distance
import numpy as np
import math

# Modified from https://github.com/TSunny007/Clustering/blob/master/notebooks/Gonzalez.ipynb
def max_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for cluster_id, cluster in enumerate(clusters):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster
            if not math.isinf(distances[point_id]):
                distances[point_id] = distances[point_id] + distance.euclidean(point,cluster)
    return data[np.argmax(distances)]

def norm_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for point_id, point in enumerate(data):
        for cluster_id, cluster in enumerate(clusters):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + math.pow(distance.euclidean(point,cluster),2)
                # return the point which is furthest away from all the other clusters
    for distance_id, current_distance in enumerate(distances):
        if not math.isinf(current_distance):
            distances[distance_id] = math.sqrt(current_distance/len(data))
    return data[np.argmax(distances)]

def gonzalez(data, cluster_num, method = 'max'):
    clusters = []
    clusters.append(data[0]) # assign the first point to the first cluster
    while len(clusters) < cluster_num:
        if method is 'max':
            clusters.append(max_dist(data, clusters))
        if method is 'norm':
            clusters.append(norm_dist(data, clusters))
        # we add the furthest point from ALL current clusters
    return (clusters)