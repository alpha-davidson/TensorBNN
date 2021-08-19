"""
An extended version of the trainRegression.py example with pretraining and
some graphs at the end to visualize the output of the BNN.
"""

import os
import math
import warnings
import time

import numpy as np
import random as rn
import tensorflow as tf
import pylab as plt


from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import GaussianDenseLayer
from tensorBNN.networkFinal import network
from tensorBNN.likelihood import FixedGaussianLikelihood
from tensorBNN.metrics import SquaredError, PercentError
from tensorBNN.predictor import predictor

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
    patience=20
    cycles=3
    epochs=100
    seed=1000


    normInfo=(0,1) # mean, sd

    #Peform pre-training to start the Markov Chain at a better spot
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(width, kernel_initializer='glorot_uniform',
                                    input_shape=(inputDims, ),
                                    activation="tanh"))
    model.add(tf.keras.layers.ReLU())

    for n in range(hidden-1):
        model.add(tf.keras.layers.Dense(width, 
                                        kernel_initializer='glorot_uniform',
                                        activation="tanh"))
        
    model.add(tf.keras.layers.Dense(outputDims, 
                                    kernel_initializer='glorot_uniform'))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=patience,
                                                restore_best_weights=True)

    #Train with decreasing learning rate
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01*(10**(-x)),
                                                         amsgrad=True),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
        model.summary()
        model.fit(trainIn, trainOut.T, validation_data=(valIn, valOut.T), 
                  epochs=epochs, batch_size=32, callbacks=[callback])

    #Save the backup 
    model.save("backup")

    #Extract weights and biases
    weights=[]
    biases=[]
    activation=[]
    for layer in model.layers:
        weightBias=layer.get_weights()
        if(len(weightBias)==2):
            weights.append(weightBias[0].T)
            bias=weightBias[1]
            bias=np.reshape(bias, (len(bias),1))
            biases.append(bias)
        if(len(weightBias)==1):
            activation.append(weightBias[0])
    
    
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
        dtype=dtype,
        weights=weights[0], biases=biases[0])
    neuralNet.add(layer) # Layer datatype
    neuralNet.add(Tanh()) # Tanh activation function
    seed += 1000 # Increment random seed
    for n in range(hidden - 1): # Add more hidden layers
        neuralNet.add(GaussianDenseLayer(width,
                                width,
                                seed=seed,
                                dtype=dtype,
                                weights=weights[n+1], biases=biases[n+1]))
        neuralNet.add(Tanh())
        seed += 1000

    neuralNet.add(GaussianDenseLayer(width,
                            outputDims,
                            seed=seed,
                            dtype=dtype,
                            weights=weights[-1], biases=biases[-1]))

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


    #Load predictor
    loadedNetwork = predictor("TrigRegression/", tf.float32)

    #Look at the predictions ins the space between the training data
    closeIn=np.linspace(-2,2,num=1000)
    closeOut = np.sin(closeIn*math.pi*2)*closeIn-np.cos(closeIn*math.pi)

    closePredictions = np.squeeze(np.array(loadedNetwork.predict(
                                                  np.array([closeIn]).T, n=1)))
    closePredictionsMean = np.mean(closePredictions, axis=0)
    closePredictionsStd = np.std(closePredictions, axis=0)
    plt.figure()

    plt.fill_between(closeIn, closePredictionsMean-2*closePredictionsStd,
                     closePredictionsMean-1*closePredictionsStd, color=(1,1,0),
                     label="2 sd")
    plt.fill_between(closeIn, closePredictionsMean-1*closePredictionsStd,
                     closePredictionsMean+1*closePredictionsStd, color=(0,1,0),
                     label="1 sd")
    plt.fill_between(closeIn, closePredictionsMean+1*closePredictionsStd,
                     closePredictionsMean+2*closePredictionsStd, color=(1,1,0))
    plt.plot(closeIn,closePredictionsMean, color="k", label="predicted mean")
    plt.plot(closeIn, closeOut, color="r", label="true")
    plt.scatter(trainIn, trainOut, color="b", label="training data")
    plt.legend()
    plt.show()

    #Look at the predictions away from the training data
    farIn=np.linspace(-4,4,num=2000)
    farOut = np.sin(farIn*math.pi*2)*farIn-np.cos(farIn*math.pi)

    farPredictions = np.squeeze(np.array(loadedNetwork.predict(
                                                    np.array([farIn]).T, n=1)))
    farPredictionsMean = np.mean(farPredictions, axis=0)
    farPredictionsStd = np.std(farPredictions, axis=0)
    
    plt.figure()
    plt.fill_between(farIn, farPredictionsMean-2*farPredictionsStd,
                     farPredictionsMean-1*farPredictionsStd, color=(1,1,0),
                     label="2 sd")
    plt.fill_between(farIn, farPredictionsMean-1*farPredictionsStd,
                     farPredictionsMean+1*farPredictionsStd, color=(0,1,0),
                     label="1 sd")
    plt.fill_between(farIn, farPredictionsMean+1*farPredictionsStd,
                     farPredictionsMean+2*farPredictionsStd, color=(1,1,0))
    plt.plot(farIn,farPredictionsMean, color="k", label="predicted mean")
    plt.plot(farIn, farOut, color="r", label="true")
    plt.scatter(trainIn, trainOut, color="b", label="training data")
    plt.legend()
    plt.show()


if(__name__ == "__main__"):
    main()
