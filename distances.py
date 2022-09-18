# file: distances.py
# author: rgreer
import numpy as np

"""
    Distance functions.
    Be careful in developing these.
    let f(x, y) be an arbitrary distance function

    assume y in R^{ input_space x n x n x ... x n }
    (where there as many n's as we
    want to reduce the space to)

    and we want f(x, y) -> R^{ n x n x ... x n }

    """

def squared_euclidean_distance(x, y):
    z = (x - y)
    return (z * z).mean(axis=0)

def euclidean_distance(x, y):
    """
        Standard Euclidean distance
    """
    return np.sqrt( squared_euclidean_distance(x, y) )
