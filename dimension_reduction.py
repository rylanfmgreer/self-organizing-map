# dimension_reduction.py
# author: rgreer

class DimensionReduction:
    """
        A generic dimension reduction object
    """
    def __init__(self, n_components):
        self._n_components = n_components