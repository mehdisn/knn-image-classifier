import numpy as np

def most_common(lst):
    """Return the most common element in a list."""
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    """Calculate Euclidean distance between a point and all points in data."""
    return np.sqrt(np.sum((point - data)**2, axis=1))
