# author: rgreer

from distances import euclidean_distance
import numpy as np


def gaussian_similarity(x, y=None):
    if y is None:
        return np.exp(x)
    else:
        d = euclidean_distance(x, y)
        return gaussian_similarity(-d)


