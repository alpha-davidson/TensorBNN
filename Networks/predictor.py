from Networks.activationFunctions import (Relu, Sigmoid, Tanh, Elu, Softmax,
                                          Leaky_relu, Prelu, SquarePrelu)
from Networks.layer import DenseLayer

import numpy as np
import tensorflow as tf


class predictor(object):
    def __init__(self, directoryPath, dtype):
        """Initializes the object

        Arguments:
            * directoryPath: file path to network folder
            * dtype: data type used for predictions
        """
        self.layerDict = {"relu": Relu, "sigmoid": Sigmoid, "tanh": Tanh,
                          "elu": Elu, "softmax": Softmax,
                          "leakyrelu": Leaky_relu, "prelu": Prelu,
                          "squareprelu": SquarePrelu, "dense": DenseLayer}

        self.directoryPath = directoryPath
        self.dtype = dtype
        self.loadNetworks()
        self.loadArchitecture()

    def loadNetworks(self):
        """Loads saved networks. """

        summary = []
        with open(self.directoryPath + "summary.txt", "r") as file:
            for line in iter(file):
                summary.append(line.split())

        numNetworks = int(summary[-1][0])
        numFiles = int(summary[-1][1])
        numMatrices = int(summary[-1][2])

        numNetworks //= numFiles

        matrices = []
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
            matrices.append(tf.cast(weights0, self.dtype))
        numNetworks *= numFiles

        self.numNetworks = numNetworks
        self.numMatrices = numMatrices
        self.matrices = matrices

    def loadArchitecture(self):
        self.layers = []

        summary = []
        with open(self.directoryPath + "architecture.txt", "r") as file:
            for line in iter(file):
                cleanedLine = line.replace("\n", "")
                self.layers.append(self.layerDict[cleanedLine](inputDims=1,
                                                               outputDims=1))

    def predict(self, inputMatrix):
        """Make predictions from an ensemble of neural networks.

        Arguments:
            * inputMatrix: The input data
        Returns:
            * initialResults: List with all predictions
        """

        inputVal = np.transpose(inputMatrix)
        initialResults = [None] * (self.numNetworks)
        for m in range(self.numNetworks):
            current = inputVal
            matrixIndex = 0
            for layer in self.layers:
                numTensors = layer.numTensors
                tensorList = []
                for x in range(numTensors):
                    tensorList.append(self.matrices[matrixIndex+x][m, :, :])
                matrixIndex += numTensors
                current = layer.predict(current, tensorList)
            initialResults[m] = current.numpy()

        return(initialResults)
