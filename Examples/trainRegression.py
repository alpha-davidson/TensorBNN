import click
import os
import time
import warnings

import numpy as np
import random as rn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from activationFunctions import SquarePrelu
from BNN_functions import normalizeData, trainBasicRegression
from layer import DenseLayer
from network import network

start = time.time()

# This supresses many deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


@click.command()
@click.option("--hidden", default=3,
              help="Number of hidden layers, default is 3")
@click.option("--width", default=50,
              help="Width of the hidden layers, default is 50")
@click.option("--epochs", default=60,
              help="Number of epochs to train for, default is 60")
@click.option("--alpha", default=0.2,
              help="Starting slope for PReLU, default is 0.2")
@click.option("--leapmin", default=500,
              help="Minimum number of leapfrog steps, default is 500")
@click.option("--leapfrog", default=750,
              help="Starting number of leapfrog steps, default is 750")
@click.option("--leapmax", default=1000,
              help="Maximum number of leapfrog steps, default is 1000")
@click.option("--leapstep", default=1,
              help="Step in leapfrog in grid search, default is 1")
@click.option("--stepmin", default=0.000005,
              help="Minimum step size, default is 0.000005")
@click.option("--stepsize", default=0.0000075,
              help="Starting step size, default is 0.0000075")
@click.option("--stepmax", default=0.000010,
              help="Maximum step size, default is 0.000010")
@click.option("--stepnum", default=100,
              help="Number of different step sizes in grid search, default" +
                   " is 100")
@click.option("--averagingsteps", default=2,
              help="Steps before param adapter suggests a new state, " +
                   "default is 2")
@click.option("--hyperstep", default=0.001,
              help="Starting step size for hyper parameters, default " +
                   "is 0.001")
@click.option("--hyperleapfrog", default=30,
              help="Number of leapfrog steps for hypers, default is 30")
@click.option("--burnin", default=0,
              help="Number of burnin epochs, default is 0")
@click.option("--increment", default=50,
              help="Epochs between saving networks, default is 50")
@click.option("--cores", default=4,
              help="Number of cores which can be used, default is 4")
@click.option("--load", default="backup",
              help="Name of a pretrained nn to load or save, default " +
                   "is backup")
@click.option("--pretrain", default=1,
              help="1 to pretrain, 2 to load pretrained, trained from" +
                   "scratch otherwise, default is 1")
@click.option(
    "--nncycles",
    default=2,
    help="Number of training cycles for normal neural net, default is 2")
@click.option(
    "--nnepochs",
    default=30,
    help="Number of epochs per neural net cycle, default is 30")
@click.option(
    "--nnpatience",
    default=30,
    help="Early stopping patience for neural net, defualt is 30")
@click.option("--name", default="Default", help="Name of network, " +
              "default is Default")
def main(
        hidden,
        width,
        epochs,
        alpha,
        leapmin,
        leapfrog,
        leapmax,
        leapstep,
        stepmin,
        stepsize,
        stepmax,
        stepnum,
        averagingsteps,
        hyperstep,
        hyperleapfrog,
        burnin,
        increment,
        cores,
        load,
        pretrain,
        nncycles,
        nnepochs,
        nnpatience,
        name):

    dtype = tf.float32
    targets = np.load("masses.npy")
    if(len(targets) == 33):
        targets = targets[1]
    else:
        targets = targets.T
        targets = targets[1]
        targets = targets.T

    inputs = np.load("massParams.npy")

    trainIn, testIn, trainOut, testOut = train_test_split(
        inputs, targets, test_size=0.20, random_state=42)

    trainIn = trainIn.T
    testIn = testIn.T
    trainOut = trainOut.T
    testOut = testOut.T

    normInfo = []
    for x in range(19):
        mean = np.mean(trainIn[x])
        sd = np.std(trainIn[x])
        normInfo.append((mean, sd))
        trainIn[x] = (trainIn[x] - mean) / sd
        testIn[x] = (testIn[x] - mean) / sd

    normInfo2 = []

    mean = np.mean(trainOut)
    sd = np.std(trainOut)
    normInfo2.append([mean, sd])
    trainOut = (trainOut - mean) / sd
    testOut = (testOut - mean) / sd

    trainIn, valIn, trainOut, valOut = train_test_split(
        trainIn.T, trainOut.T, test_size=0.20, random_state=21)

    trainIn = tf.cast(trainIn, dtype)
    valIn = tf.cast(valIn, dtype)
    trainOut = tf.cast(trainOut, dtype)
    valOut = tf.cast(valOut, dtype)
    data = [trainIn, trainOut, valIn, valOut]

    # 19 input values and 1 output
    outputDims = 1
    inputDims = 19

    weights = [None] * hidden
    biases = [None] * hidden
    activation = [None] * hidden
    if(pretrain == 1):
        if(nncycles > 0):
            weights, biases, activation = \
                    trainBasicRegression(hidden, inputDims, outputDims, width,
                                         nncycles, nnepochs, alpha, nnpatience,
                                         data[0], data[1], data[2], data[3],
                                         load)

    if(pretrain == 2):
        model = tf.keras.models.load_model(load)
        weights = []
        biases = []
        activation = []
        for layer in model.layers:
            weightBias = layer.get_weights()
            if(len(weightBias) == 2):
                weights.append(tf.cast(tf.transpose(weightBias[0]), dtype))
                bias = weightBias[1]
                bias = np.reshape(bias, (len(bias), 1))
                biases.append(tf.cast(bias, dtype))
            if(len(weightBias) == 1):
                activation.append(tf.cast((weightBias[0]), dtype))

    normInfo2 = np.array(normInfo2).T

    neuralNet = network(
        dtype, inputDims, data[0], data[1], data[2], data[3], tf.cast(
            normInfo2[0], dtype), tf.cast(
            normInfo2[1], dtype))

    # Add the layers
    seed = 0
    neuralNet.add(
        DenseLayer(
            inputDims,
            width,
            weights=weights[0],
            biases=biases[0],
            seed=seed,
            dtype=dtype))
    neuralNet.add(SquarePrelu(width, alpha=alpha**(0.5), activation=None,
                  dtype=dtype))
    seed += 1000
    for n in range(hidden - 1):
        neuralNet.add(DenseLayer(width,
                                 width,
                                 weights=weights[n + 1],
                                 biases=biases[n + 1],
                                 seed=seed,
                                 dtype=dtype))
        neuralNet.add(
            SquarePrelu(
                width,
                alpha=alpha**(0.5),
                activation=None,
                dtype=dtype))
        seed += 1000
    neuralNet.add(DenseLayer(width,
                             outputDims,
                             weights=weights[-1],
                             biases=biases[-1],
                             seed=seed,
                             dtype=dtype))

    # Setup the markov chain monte carlo
    neuralNet.setupMCMC(
        stepsize,
        stepmin,
        stepmax,
        stepnum,
        leapfrog,
        leapmin,
        leapmax,
        leapstep,
        hyperstep,
        hyperleapfrog,
        burnin,
        cores,
        averagingsteps)

    # Train the network
    neuralNet.train(
        epochs,
        burnin,
        increment,
        scaleExp=False,
        folderName=name,
        networksPerFile=25,
        returnPredictions=False,
        regression=True)

    end = time.time()
    print()
    print("elapsed time:", end - start)


if(__name__ == "__main__"):
    main()
