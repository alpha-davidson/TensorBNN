import os
import math
import warnings
import time

import numpy as np
import random as rn
import tensorflow as tf

from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import GaussianDenseLayer
from tensorBNN.networkFinal import network
from tensorBNN.likelihood import FixedGaussianLikelihood
from tensorBNN.metrics import SquaredError, PercentError

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

    trainIn=np.linspace(-2,2,num=11)
    valIn=np.linspace(-2+2/30,2.0-2/30,num=30)
    trainOut = np.sin(trainIn*math.pi*2)*trainIn-np.cos(trainIn*math.pi)
    valOut = np.sin(valIn*math.pi*2)*valIn-np.cos(valIn*math.pi)


    data=[trainIn, trainOut, valIn, valOut]

    dtype=tf.float32
    
    inputDims=1
    outputDims=1
    width = 10 # perceptrons per layer
    hidden = 3 # number of hidden layers
    seed=1000
    
    inputDims=1
    outputDims=1

    normInfo=(0,1) # mean, sd

    likelihood=FixedGaussianLikelihood(sd=0.1)
    metricList=[SquaredError(mean=normInfo[0], sd=normInfo[1]), 
                PercentError(mean=normInfo[0], sd=normInfo[1])]

    neuralNet = network(
                dtype, # network datatype
                inputDims, # dimension of input vector
                data[0], # training input data
                data[1].T, # training output data
                data[2], # validation input data
                data[3].T) # validation output data)
    
    layer = GaussianDenseLayer( # Dense layer object
        inputDims, # Size of layer input vector
        width, # Size of layer output vector
        seed=seed, # Random seed
        dtype=dtype)
    neuralNet.add(layer) # Layer datatype
    neuralNet.add(Tanh()) # Tanh activation function
    seed += 1000 # Increment random seed
    for n in range(hidden - 1): # Add more hidden layers
        neuralNet.add(GaussianDenseLayer(width,
                                width,
                                seed=seed,
                                dtype=dtype))
        neuralNet.add(Tanh())
        seed += 1000

    neuralNet.add(GaussianDenseLayer(width,
                            outputDims,
                            seed=seed,
                            dtype=dtype))

    neuralNet.setupMCMC(
        stepSizeStart=1e-3,#0.0004 # starting stepsize
        stepSizeMin=1e-4, #0.0002 # minimum stepsize
        stepSizeMax=1e-2, # maximum stepsize
        stepSizeOptions=100, # number of stepsize options in stepsize adapter
        leapfrogStart=1000, # starting number of leapfrog steps
        leapfogMin=100, # minimum number of leapfrog steps
        leapFrogMax=10000, # maximum number of leapfrog steps
        leapfrogIncrement=10, # stepsize between leapfrog steps in leapfrog step adapter
        hyperStepSize=0.001, # hyper parameter stepsize
        hyperLeapfrog=100, # hyper parameter number of leapfrog steps
        burnin=1000, # number of burnin epochs
        averagingSteps=10) # number of averaging steps for param adapters)

		
    neuralNet.train(
        6001, # epochs to train for
        10, # increment between network saves
        likelihood,
        metricList=metricList,
        adjustHypers=True,
        folderName="TrigRegression", # Name of folder for saved networks
        networksPerFile=50) # Number of networks saved per file
        
    print("Total time elapsed (seconds):", time.time() - startTime)



if(__name__ == "__main__"):
    main()
