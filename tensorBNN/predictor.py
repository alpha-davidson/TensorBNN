from tensorBNN.activationFunctions import (Relu, Sigmoid, Tanh, Elu, Softmax,
                                          Leaky_relu, Prelu, SquarePrelu)
from tensorBNN.layer import DenseLayer
from tensorBNN.likelihood import GaussianLikelihood

from emcee.autocorr import integrated_time, function_1d

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import math


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

    def predict(self, inputMatrix, n=1):
        """Make predictions from an ensemble of neural networks.

        Arguments:
            * inputMatrix: The input data
            * n: Predict using every n networks
        Returns:
            * initialResults: List with all networks used
        """

        inputVal = np.transpose(inputMatrix)
        initialResults = [None] * math.ceil(self.numNetworks/n)
        for m in range(0, self.numNetworks, n):
            current = inputVal
            matrixIndex = 0
            for layer in self.layers:
                numTensors = layer.numTensors
                tensorList = []
                for x in range(numTensors):
                    tensorList.append(self.matrices[matrixIndex+x][m, :, :])
                matrixIndex += numTensors
                current = layer.predict(current, tensorList)
            initialResults[m//n] = current.numpy()

        return(initialResults)
    
    def trainProbs(self, trainX, trainY, n, likelihood):
        """ Calculate the negative log likelihoods for the training data.
        
        Arguments:
            * trainX: training input data
            * trainY: training output data
            * n: Predict using every n networks
        
        """
        weights = []
        if(likelihood is not None):
            weights = self.likelihood.calcultateLogProb(input=tf.transpose(trainX),
                                                    realVals=trainY,
                                                    n=n,
                                                    hypers = self.hypers,
                                                    predict = self.predict,
                                                    dtype=self.dtype)
        else:
            for m in range(0, self.numNetworks, n):
                weights.append(tf.cast(0, self.dtype))
        for m in range(0, self.numNetworks, n):
            matrixIndex = 0
            hyperIndex = 0
            current = -weights[m//n]
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
            weights[m//n] = current
        self.weightsTrain = np.array(weights)
   
    def reweight(self, architecture, trainX=None, trainY=None, n=1, likelihood=None):
        """ Calculate new weights for each network if they have the new 
        hyper paramters described in architecture. The weights are calculated
        according to p(theta|priors2)/p(theta|priors1). The new priors can be
        anything, but the layers must still accept the same size inputs and 
        number of hyper paramters as the base networks.
        
        Arguments:
            * trainX: training input data
            * trainY: training output data
            * architecture: new architecture file
            * n: Predict using every n networks
        
        Returns:
            * weighting: Numpy array with new weights for the networks.
        """
        
        if(len(self.weightsTrain)==0):
            self.trainProbs(trainX, trainY, n, likelihood)
        
        self.loadArchitecture(architecture=architecture)
        
        weights = []
        if(likelihood is not None):
            weights = self.likelihood.calcultateLogProb(input=tf.transpose(trainX),
                                                    realVals=trainY,
                                                    n=n,
                                                    hypers = self.hypers,
                                                    predict = self.predict,
                                                    dtype=self.dtype)
        else:
            for m in range(0, self.numNetworks, n):
                weights.append(tf.cast(0, self.dtype))

        for m in range(0, self.numNetworks, n):
            matrixIndex = 0
            hyperIndex = 0
            current = -weights[m//n]
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
            weights[m//n] = current
        self.weights = np.array(weights)
        
        weighting = np.exp(self.weightsTrain-weights)
        weighting/=np.sum(weighting)
        
        self.loadArchitecture()
        
        return(weighting)

    def autocorrelation(self, inputData, nMax):
        predictions = self.predict(inputData, n=1)
        output = np.squeeze(np.array(predictions)).T
        
        valFunc=0
        accepted=0
        
        for x in range(len(output)):
            temp = (integrated_time(output[x], tol=5, quiet=True))
            if(not math.isnan(temp)):
                valFunc += np.array((function_1d(output[x])))
                accepted+=1
        
        valFunc=valFunc/accepted
        if(nMax<len(valFunc)):
            valFunc = valFunc[:nMax]
        
        return(valFunc)
        
    def autoCorrelationLength(self, inputData, nMax):
        predictions = self.predict(inputData, n=1)
        output = np.squeeze(np.array(predictions)).T
        
        val=0
        accepted=0
        
        for x in range(len(output)):
            temp = (integrated_time(output[x], tol=5, quiet=True))
            if(not math.isnan(temp)):
                val += temp
                accepted+=1
        
        val=val/accepted
        
        if(val[0]>nMax):
            print("Correlation time is greater than maximum accepted value.")
        
        return(val[0])
