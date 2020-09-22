from TensorBNN.activationFunctions import (Relu, Sigmoid, Tanh, Elu, Softmax,
                                          Leaky_relu, Prelu, SquarePrelu)
from TensorBNN.layer import DenseLayer
from TensorBNN.likelihood import GaussianLikelihood

import numpy as np
import tensorflow as tf


class predictor(object):
    def __init__(self, directoryPath, dtype, customLayerDict={},
                 likelihood=GaussianLikelihood(sd = 0.1)):
        """ The constructor here obtains the necesary information to make basic
        predictions, and also the basic likelihood function for future
        reweighting.
        
        Arguments:
            * directoryPath: Path to folder containing saved networks
            * dtype: data type of network
            * customLayerDict: Dictionary containing any custom layers with
            their names as keys
            * likelihood: Likelihood object used in training. If reweighting
            is not performed this does not matter.
        
        """
        self.layerDict = {"relu": Relu, "sigmoid": Sigmoid, "tanh": Tanh,
                          "elu": Elu, "softmax": Softmax,
                          "leakyrelu": Leaky_relu, "prelu": Prelu,
                          "squareprelu": SquarePrelu, "dense": DenseLayer}
        self.directoryPath = directoryPath
        self.layerDict.update(customLayerDict)
        self.dtype = dtype
        self.loadNetworks()
        self.loadArchitecture()
        self.likelihood = likelihood
        self.weightsTrain = []

    def loadNetworks(self):
        """Loads saved networks.

        """

        summary = []
        with open(self.directoryPath + "summary.txt", "r") as file:
            for line in iter(file):
                summary.append(line.split())
        numNetworks = int(summary[-2][0])
        numFiles = int(summary[-2][1])
        numMatrices = int(summary[-2][2])
        numHypers = int(summary[-1][0])
        
        numNetworks //= numFiles

        matrices = []
        vectors=[]
        for x in range(numFiles*numNetworks):
            vectors.append([])
        
        for n in range(numMatrices):
            if(len(summary[n]) == 2):
                weightsSplitDims = (numNetworks *
                                    numFiles, int(summary[n][0]),
                                    int(summary[n][1]))
            else:
                weightsSplitDims = (numNetworks *
                                    numFiles, int(summary[n][0]), int(1))
            weights0 = np.zeros(weightsSplitDims)
            for m in range(numFiles):
                weights = np.loadtxt(
                    self.directoryPath +
                    str(n) +
                    "." +
                    str(m) +
                    ".txt",
                    dtype=np.float32,
                    ndmin=2)
                for k in range(numNetworks):
                    
                    netNumber = m * numNetworks + k
                    index1 = weightsSplitDims[1] * k
                    index2 = weightsSplitDims[1] * (k + 1)
                    index3 = weightsSplitDims[2]
                    weights0[netNumber, :, :] = weights[index1:index2, :index3]
                    vectors[netNumber].append(tf.cast(weights[index1:index2, :index3], self.dtype).numpy().flatten())
                    
            matrices.append(tf.cast(weights0, self.dtype))
        print(len(vectors), len(vectors[0]))
        for x in range(len(vectors)):
            vectors[x] = np.concatenate(vectors[x])
        
        hypers=[]
        if(numHypers>0):
            for m in range(numFiles):
                weights = np.loadtxt(
                    self.directoryPath + "hypers" + str(m) + ".txt",
                    dtype=np.float32, ndmin=1)
                for k in range(numNetworks):
                    netNumber = m * numNetworks + k
                    index1 = numHypers * k
                    index2 = numHypers * (k + 1)
                    hypers.append(weights[index1:index2])
        
        
        
        numNetworks *= numFiles

        self.numNetworks = numNetworks
        self.numMatrices = numMatrices
        self.matrices = matrices
        self.hypers=hypers
        self.vectors=vectors
        
    def loadArchitecture(self, architecture=None):
        self.layers = []
        if(architecture is None):
            with open(self.directoryPath + "architecture.txt", "r") as file:
                for line in iter(file):
                    cleanedLine = line.replace("\n", "")
                    self.layers.append(self.layerDict[cleanedLine](inputDims=1,
                                                                   outputDims=1))
        else:
            with open(architecture, "r") as file:
                for line in iter(file):
                    cleanedLine = line.replace("\n", "")
                    self.layers.append(self.layerDict[cleanedLine](inputDims=1,
                                                                   outputDims=1))

    def predict(self, inputMatrix, skip=10):
        """Make predictions from an ensemble of neural networks.

        Arguments:
            * inputMatrix: The input data
            * skip: Predict using every n networks where n = skip
        Returns:
            * initialResults: List with all networks used
        """

        inputVal = np.transpose(inputMatrix)
        initialResults = [None] * (self.numNetworks//skip)
        for m in range(0, self.numNetworks, skip):
            current = inputVal
            matrixIndex = 0
            for layer in self.layers:
                numTensors = layer.numTensors
                tensorList = []
                for x in range(numTensors):
                    tensorList.append(self.matrices[matrixIndex+x][m, :, :])
                matrixIndex += numTensors
                current = layer.predict(current, tensorList)
            initialResults[m//skip] = current.numpy()

        return(initialResults)
    
    def trainProbs(self, trainX, trainY, skip):
        """ Calculate the negative log likelihoods for the training data.
        
        Arguments:
            * trainX: training input data
            * trainY: training output data
            * skip: Predict using every n networks where n = skip
        
        """
        
        weights = self.likelihood.calcultateLogProb(input=tf.transpose(trainX),
                                                    realVals=trainY,
                                                    skip=skip,
                                                    hypers = self.hypers,
                                                    predict = self.predict,
                                                    dtype=self.dtype)

        for m in range(0, self.numNetworks, skip):
            matrixIndex = 0
            hyperIndex = 0
            current = -weights[m//skip]
            for layer in self.layers:
                numTensors = layer.numTensors
                numHyperTensors = layer.numHyperTensors
                tensorList = []
                hyperList = []
                for x in range(numTensors):
                    tensorList.append(self.matrices[matrixIndex+x][m, :, :])
                for x in range(numHyperTensors):
                    hyperList.append(self.hypers[m][hyperIndex+x])
                hyperIndex += numHyperTensors
                matrixIndex += numTensors
                current -= tf.cast(layer.calculateHyperProbs(hyperList, tensorList), self.dtype).numpy()
            weights[m//skip] = current
        self.weightsTrain = np.array(weights)
   
    def reweight(self, trainX, trainY, architecture, skip=10):
        """ Calculate new weights for each network if they have the new 
        hyper paramters described in architecture. The weights are calculated
        according to p(theta|priors2)/p(theta|priors1). The new priors can be
        anything, but the layers must still accept the same size inputs and 
        number of hyper paramters as the base networks.
        
        Arguments:
            * trainX: training input data
            * trainY: training output data
            * architecture: new architecture file
            * skip: Predict using every n networks where n = skip
        
        Returns:
            * weighting: Numpy array with new weights for the networks.
        """
        
        if(len(self.weightsTrain)==0):
            self.trainProbs(trainX, trainY, skip)
        
        self.loadArchitecture(architecture=architecture)
        
        weights = self.likelihood.calcultateLogProb(input=tf.transpose(trainX),
                                                    realVals=trainY,
                                                    skip=skip,
                                                    hypers = self.hypers,
                                                    predict = self.predict,
                                                    dtype=self.dtype)

        for m in range(0, self.numNetworks, skip):
            matrixIndex = 0
            hyperIndex = 0
            current = -weights[m//skip]
            for layer in self.layers:
                numTensors = layer.numTensors
                numHyperTensors = layer.numHyperTensors
                tensorList = []
                hyperList = []
                for x in range(numTensors):
                    tensorList.append(self.matrices[matrixIndex+x][m, :, :])
                for x in range(numHyperTensors):
                    hyperList.append(self.hypers[m][hyperIndex+x])
                hyperIndex += numHyperTensors
                matrixIndex += numTensors
                current -= tf.cast(layer.calculateHyperProbs(hyperList, tensorList), self.dtype).numpy()
            weights[m//skip] = current
        self.weights = np.array(weights)
        
        weighting = np.exp(self.weightsTrain-weights)
        weighting/=np.sum(weighting)
        
        self.loadArchitecture()
        
        return(weighting)

    def correlation(self, skip=1):
        """ Calcualte the Pearson product-moment correlation between networks.
        
        Arguments:
            * skip: Predict using every n networks where n = skip
            
        Returns:
            * coef: The coefficients between the networks used
        
        """
        coef=[]
        for m in range(0, self.numNetworks-skip, skip):
            print()
            matrix = np.array([self.vectors[m], self.vectors[m+skip]])
            print(np.corrcoef(matrix))
            
            coef.append(np.corrcoef(matrix)[0,1])
        return(coef)