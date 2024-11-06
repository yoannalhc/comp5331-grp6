import random
import numpy as np

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def carve(points, R, k):
    centers = []
    uncovered_indices = set(range(len(points)))  # Indices of uncovered points

    while uncovered_indices and len(centers) < k:
        # Randomly select an uncovered point
        idx = random.choice(list(uncovered_indices))
        center = points[idx]
        centers.append(center)

        # Mark all points within distance R from the new center as covered
        to_remove = []
        for i in uncovered_indices:
            if distance(center, points[i]) <= R:
                to_remove.append(i)

        # Remove covered points from uncovered set
        uncovered_indices.difference_update(to_remove)

    return centers

def find_minimum_R(points, k, R_start, R_end, step=0.1):
    best_R = None

    R = R_start
    while R <= R_end:
        centers = carve(points, R, k)
        if len(centers) <= k:  # Check if we opened at most k centers
            best_R = R  # Update best R found
            R -= step  # Try a smaller R
        else:
            R += step  # Increase R
    return best_R