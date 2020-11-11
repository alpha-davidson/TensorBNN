# Regression Example
Here, I will present an example of using `tensorBNN` to train a very basic regression problem. It will also highlight how the BNN represents model uncertainty very well. 
First, we need to import the nescesary packages. This is done through the commands
## Program Setup
```
import os
import math

import numpy as np
import random as rn
import tensorflow as tf

from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import DenseLayer
from tensorBNN.network import network
from tensorBNN.likelihood import GaussianLikelihood
```
In order to obtain reproducible results we need to set random seeds. In order to be sure that absolutely everything is seeded, we use the following four lines of code
```
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)
```
# Data preparation 
Next, we need to generate our dataset. We are simply going to use the function ```f(x)=x*sin(2pi*x)-cos(pi*x).``` 
We will generate a training dataset with 31 points and a validation dataset with 30 points. This is done as follows.
```
trainIn=np.linspace(-2,2,num=31)
valIn=np.linspace(-2+2/30,2.0-2/30,num=30)
trainOut = np.sin(trainIn*math.pi*2)*trainIn-np.cos(trainIn*math.pi)
valOut = np.sin(valIn*math.pi*2)*valIn-np.cos(valIn*math.pi)
```
After this we need to group our data together and declare the dataype we will be using.
```
data=[trainIn, trainOut, valIn, valOut]

dtype=tf.float32
```
## Network setup
To get the network setup we need to first declare the number of input and output dimensions and the normalization we used on our output data. As we didn't normalize, we just say we have a mean of 0 and a standard deviation of 0 so `TensorBNN` doesn't try and unnormalize the data.
```
inputDims=1
outputDims=1

normInfo=(0,1) # mean, sd
```
Now we actually need to create the network object. This is done like so.
```
neuralNet = network(
            dtype, # network datatype
            inputDims, # dimension of input vector
            data[0], # training input data
            data[1], # training output data
            data[2], # validation input data
            data[3], # validation output data)
```
Next, we add the layers. We will be using two hidden layers with 10 perceptrons each and the hyperbolic tangent activation function.
```
width = 10 # perceptrons per layer
hidden = 2 # number of hidden layers
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
    neuralNet.add(Tanh())
    seed += 1000

neuralNet.add(DenseLayer(width,
                         outputDims,
                         seed=seed,
                         dtype=dtype))
```
Now we need to initialize the Markov Chain Monte Carlo algorithm. We do this with the following code.
```
neuralNet.setupMCMC(
        0.005, # starting stepsize
        0.0025, # minimum stepsize
        0.01, # maximum stepsize
        40, # number of stepsize options in stepsize adapter
        2, # starting number of leapfrog steps
        2, # minimum number of leapfrog steps
        50, # maximum number of leapfrog steps
        1, # stepsize between leapfrog steps in leapfrog step adapter
        0.01, # hyper parameter stepsize
        5, # hyper parameter number of leapfrog steps
        20, # number of burnin epochs
        20, # number of cores
        2) # number of averaging steps for param adapters)
```
Next we initialize the Likelihood object we use to evaluate predictions. We use a Gaussian likelihood with a starting standard deviation of 0.1.
```
likelihood = GaussianLikelihood(sd = 0.1)
```
We would also like to measure the performance of the network using a metric such as mean squared error, so we initialize a metric object and add it to a metric list.
```
metricList = [SquaredError()]
```

Finally, we get to actually train the network. This is done with the following code.
```
neuralNet.train(
        1000, # epochs to train for
        2, # increment between network saves
        metricList = metricList, # List of evaluation metricx
        folderName="TrigRegression") # Name of folder for saved networks
```
After this, just run the program. 
