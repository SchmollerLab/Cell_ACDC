import numpy as np

def nearest_ID_to_centroid(a, y, x, max_iterations=10, distance_threshold=5):
    """
    Return cell ID by checking `max_iterations` nearest non-zero pixels
    and seeing if their object centroid is within a certain distance
    from the given coordinates (y, x).
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)  # Ensure a is a numpy array
        
    r, c = np.nonzero(a)
    if r.size == 0:
        return None
    
    distances = np.linalg.norm(np.array([r, c]).T - np.array([y, x]), axis=1)
    sorted_indices = np.argsort(distances)
    sorted_IDs = a[r, c][sorted_indices]
    # Remove duplicates, preserving order
    _, unique_indices = np.unique(sorted_IDs, return_index=True)
    sorted_unique_IDs = sorted_IDs[np.sort(unique_indices)]

    for target_ID in sorted_unique_IDs[:max_iterations]:
        coords = np.where(a == target_ID)
        centroid = np.mean(coords, axis=1)
        centroid_dist = np.linalg.norm(centroid - np.array([y, x]))
        if centroid_dist <= distance_threshold:
            return target_ID

    return None  # If no ID found within the distance threshold after max_iterations
