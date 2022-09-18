import numpy as np
from similarities import gaussian_similarity
from distances import squared_euclidean_distance

class NeighbourhoodFunction:
    """
        https://www.cs.bham.ac.uk/~jxb/NN/l16.pdf
    """
    def __init__(self, sigma_0, tau_n):
        self.sigma_0 = sigma_0
        self.tau_n = tau_n

    def evaluate(self, target, full_map, t):
        s_ed = squared_euclidean_distance(target, full_map)
        sigma = 2.0 * self.sigma_0 * np.exp(-t / self.tau_n)
        return np.exp(-s_ed / sigma)

    def __call__(self, target, full_map, t):
        return self.evaluate(target, full_map, t)


class LearningRate:
    """
        learning rate 
    """
    def __init__(self, initialLR, tau_n):
        self.initialLR = initialLR
        self.tau_n = tau_n
        self.iteration = 0

    def evaluate(self):
        rateToReturn = np.exp(-self.iteration / self.tau_n) * self.initialLR
        self.iteration += 1
        return rateToReturn

    def __call__(self):
        return self.evaluate()
    
    def reset(self):
        self.iteration = 0