import os
import math
import warnings
import time

import numpy as np
import random as rn
import tensorflow as tf

from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import DenseLayer
from tensorBNN.network import network
from tensorBNN.likelihood import GaussianLikelihood
from tensorBNN.metrics import SquaredError, PercentError

import time

startTime = time.time()

# This supresses many deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


def main():

    trainIn=np.linspace(-2,2,num=31)
    valIn=np.linspace(-2+2/30,2.0-2/30,num=30)
    trainOut = np.sin(trainIn*math.pi*2)*trainIn-np.cos(trainIn*math.pi)
    valOut = np.sin(valIn*math.pi*2)*valIn-np.cos(valIn*math.pi)


    data=[trainIn, trainOut, valIn, valOut]

    dtype=tf.float32

    inputDims=1
    outputDims=1

    normInfo=(0,1) # mean, sd

    likelihood=GaussianLikelihood(sd=0.1)
    metricList=[SquaredError(mean=normInfo[0], sd=normInfo[1]), PercentError(mean=normInfo[0], sd=normInfo[1])]

    neuralNet = network(
                dtype, # network datatype
                inputDims, # dimension of input vector
                data[0], # training input data
                data[1], # training output data
                data[2], # validation input data
                data[3]) # validation output data

       width = 10 # perceptrons per layer
    hidden = 3 # number of hidden layers
    seed = 0 # random seed
    neuralNet.add(
        DenseLayer( # Dense layer object
            inputDims, # Size of layer input vector
            width, # Size of layer output vector
            seed=seed, # Random seed
            dtype=dtype)) # Layer datatype
    neuralNet.add(Tanh()) # Tanh activation function
    seed += 1000 # Increment random seed
    for n in range(hidden - 1): # Add more hidden layers
        neuralNet.add(DenseLayer(width,
                                 width,
                                 seed=seed,
                                 dtype=dtype))
        #neuralNet.add(Relu())
        neuralNet.add(Tanh())
        seed += 1000

    neuralNet.add(DenseLayer(width,
                             outputDims,
                             seed=seed,
                             dtype=dtype))

    neuralNet.setupMCMC(
        0.005, # starting stepsize
        0.0025, # minimum stepsize
        0.01, # maximum stepsize
        40, # number of stepsize options in stepsize adapter
        2, # starting number of leapfrog steps
        2, # minimum number of leapfrog steps
        50, # maximum number of leapfrog steps
        1, # stepsize between leapfrog steps in leapfrog step adapter
        0.0015, # hyper parameter stepsize
        5, # hyper parameter number of leapfrog steps
        20, # number of burnin epochs
        20, # number of cores
        2) # number of averaging steps for param adapters)
		
    neuralNet.train(
        1020, # epochs to train for
        10, # increment between network saves
        likelihood,
        metricList=metricList,
        folderName="TrigRegression", # Name of folder for saved networks
        networksPerFile=50) # Number of networks saved per file
        
    print("Time elapsed:", time.time() - startTime)


if(__name__ == "__main__"):
    main()
