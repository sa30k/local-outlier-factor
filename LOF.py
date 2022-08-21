import numpy as np

def get_distance_matrix(X, Y):
    """
    Compute the Euclidean distance matrix between X and Y
    
    Parameters
    ----------
    X : ndarray
    Y : ndarray

    Returns
    -------
    dist : ndarray
    """

    X = X[:, np.newaxis, :]
    Y = Y[np.newaxis, :, :]
    dist = np.sqrt(np.sum((X-Y)**2, axis=2))
    return dist

def get_k_nearest_neighbors(data, k):
    """
    Compute the k-nearest distances and corresponding indexes for each data point
    
    Parameters
    ----------
    data : ndarray
    k : int

    Returns
    -------
    k_nearest_dist: ndarray
    k_nearest_idx: ndarray
    """

    dist = get_distance_matrix(data, data)

    k_nearest_idx = np.argsort(dist, axis=1)[:, 1:k+1]
    k_nearest_dist = np.sort(dist, axis=1)[:, 1:k+1]

    return k_nearest_dist, k_nearest_idx

def get_local_reachability_density(data, k):
    """
    Compute the local reachability density for each data point
    
    Parameters
    ----------
    data : ndarray
    k : int

    Returns
    -------
    local_reachability_density: ndarray
    k_nearest_idx: ndarray
    """

    k_nearest_dist, k_nearest_idx = get_k_nearest_neighbors(data, k)

    kth_nearest_dist_of_neighbors = k_nearest_dist[k_nearest_idx, k-1]
    reachability_distance = np.maximum(k_nearest_dist, kth_nearest_dist_of_neighbors)

    local_reachability_density = k / np.sum(reachability_distance, axis=1)

    return local_reachability_density, k_nearest_idx

def get_local_outlier_factor(data, k):
    """
    Compute the local outlier factor for each data point
    
    Parameters
    ----------
    data : ndarray
    k : int

    Returns
    -------
    LOF: ndarray
    """
    local_reachability_density, k_nearest_idx = get_local_reachability_density(data, k)

    neighbors_lrd = local_reachability_density[k_nearest_idx]
    LOF = np.sum(neighbors_lrd, axis=1) / (k * local_reachability_density)

    return LOF