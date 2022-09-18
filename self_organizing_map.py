# self_organizing_map.py
# author: rgreer
# for me to learn this dimension reduction technique :-)

import numpy as np
from dimension_reduction import DimensionReduction
from distances import *
from similarities import *
from sklearn.decomposition import PCA
from utility_classes import NeighbourhoodFunction, LearningRate
from utility_functions import create_fake_data, check_SOM

class SelfOrganizingMap(DimensionReduction):
    """
        A Self Organizing Map for dimension reduction.
        Rectangular.

        TODO: put more of the member functions stuff in the
        initialization
    """
    def __init__(self,
                n_components=2,
                learning_rate=1,
                n_iterations=10_000,
                square_size=10,
                beta=0.01,
                beta_decay=100_000,
                initialization=PCA,
                sigma_0=10
        ):
                
        DimensionReduction.__init__(self, n_components=n_components)
        self._distance    = euclidean_distance
        self._lr          = learning_rate
        self._n_iter      = n_iterations
        self._square_size = square_size
        self.initialization = initialization(n_components=n_components)
        self._is_fit = False

        self.learningRate = LearningRate(beta, beta_decay)
        self.neighbourhoodFunction = NeighbourhoodFunction(
            sigma_0, beta_decay)


    def _initializeMap(self, inputSpaceDimension, X=None):
        if not isinstance(self.initialization, PCA):
            raise NotImplementedError

        # get initial fit
        self.initialization.fit(X.T)
        components = self.initialization.components_
        
        # initialize the map as different
        # linear combinations of the components
        map_components = np.meshgrid(
            *[np.linspace(-10, 10, self._square_size)
                for i in range(self._n_components)])

        self._map = np.tensordot(components[0],
                map_components[0], axes=0)
        for i in range(1, self._n_components):
            self._map += np.tensordot(components[i],
                    map_components[i], axes=0)



    def _getInputVector(self, X):
        inputColIdx = np.random.randint(X.shape[1])
        inputVector = X[:, inputColIdx] 
        return inputVector


    def _normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)


    def _reshapeInputVector(self, inputVector):
        newDims = [self.nInputDims] + [1] * self._n_components
        newInput = inputVector.reshape(newDims)
        return newInput


    def _calcAllDistances(self, inputVector):
        inputVector = self._reshapeInputVector(inputVector)
        return self._distance(inputVector, self._map)


    def _getBMUIndex(self, distances):
        flatIndex = distances.argmin()
        tupleIndex = np.unravel_index(flatIndex,
                        distances.shape)
        return tupleIndex


    def _calcAllSimilarities(self, BMUIndex,
            minDist=1e-4, cutDist=False, pctleAlpha=0.8,
            n=3):
        # calculates similarities between
        # all the weight vectors in the map, 
        # and the weight vector located at BMUIndex

        map_dummy = np.moveaxis(self._map, 0, -1)
        v = map_dummy[BMUIndex]
        v = self._reshapeInputVector(v)
        sims = self.neighbourhoodFunction(v, self._map, self.iteration)

        # only include those that are actually close
        bound = np.percentile(sims, pctleAlpha * 100)
        sims = np.where(sims <  bound, 0., sims)    

        return sims


    def _updateWeights(self, similarities, inputVector):
        
        # can we take this calculation from somewhere?
        reshapedInput = self._reshapeInputVector(inputVector)
        reshapedInput = np.broadcast_to(reshapedInput, self._map.shape)
        movement = reshapedInput - self._map
        update = movement * similarities * self.learningRate()
        self._map += update


    def _updateParams(self, i):
        # deprecated but keeping just in case
        pass
            

    def fit(self, X):
        """
            Fit the self organizing map.
            Assumes an observation of X is a column
        """
        self.nInputDims, self.nObs = X.shape
        X = self._normalize(X)
        self._initializeMap(self.nInputDims, X)
        self.iteration = 0

        for i in range(self._n_iter):
            inputVector = self._getInputVector(X)
            distances = self._calcAllDistances(inputVector)
            BMUIdx = self._getBMUIndex(distances)
            similarities = self._calcAllSimilarities(BMUIdx)
            self._updateWeights(similarities, inputVector)
            self._updateParams(i)
            self.iteration = i

        print('Fit complete.')

    def evaluate(self, X):
        nObsToTransform = X.shape[1]
        returnArr = np.zeros( (self._n_components, nObsToTransform))
        X = self._normalize(X)
        
        if not self._is_fit:
            self.nInputDims, self.nObs = X.shape
            self._initializeMap(self.nInputDims, X)

        for i in range( nObsToTransform ):
            inputVector = X[:, i]
            distances = self._calcAllDistances(inputVector)
            BMUIdx = np.array(self._getBMUIndex(distances))
            returnArr[:, i] = BMUIdx
        return returnArr


if __name__ == '__main__':
    SOM = SelfOrganizingMap( n_iterations=10000, square_size=20,
    beta=0.2,)
    X, y = create_fake_data()
    check_SOM(SOM, X, y)
    SOM.fit(X)
    finished = True
    check_SOM(SOM, X, y)
    really_finished=True