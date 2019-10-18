# MNIST ClassificationExample
On this page is a tutorial on training a classification BNN using the tools available inside of `TensorBNN` using the `MNIST` dataset. This dataset consists of a collection of 28x28 grayscale images of handwritten digits. This tutorial will show how to select two numbers and train a BNN to detect between the two.

## Data setup
First, it is necesary to import all the packages that will be needed. The required ones are
```
import os

import numpy as np
import random as rn
import tensorflow as tf

from sklearn.model_selection import train_test_split

from Networks.activationFunctions import SquarePrelu, Sigmoid
from Networks.BNN_functions import trainBasicClassification
from Networks.layer import DenseLayer
from Networks.network import network
```
The 'os', 'numpy', 'random', and 'tensorflow' imports are all required to set the random seeds properly so that results are reproducible. The other imports are either for training and validation data spliting or for constructing the actual network. It is important to note, however, that if a GPU is used for training, which is highly recomended, it is impossible to obtain completely reproducible results simply because of how a GPU works. The code required to set these random seeds is:

```
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)
```
After setting up the random seeds, we need to get our dataset. This is accomplished though the code: 
``` 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
```
As the `MNIST` data consists of a bunch of pictures we need to reshape each picture into a vector and scale the pixel values between 0 and 1. This is accomplished here:
```
x_train_shape = x_train.shape

inputDims=x_train_shape[1]**2
outputDims=1

x_train = np.reshape(x_train, (x_train_shape[0],x_train_shape[1]**2))
x_train = np.float32(x_train)/256
```
We also collected our input and output dimensions, which will be important later.
Next, we must collect the two numbers that we are interested in. For this tutorial we will use 3 and 8, but you are free to use whatever two numbers you desire. In the following block of code we create our new datasets.
```
new_x_train = []
new_y_train = []

for y in range(len(y_train)):
    if(y_train[y]==3):
        new_y_train.append(0)
        new_x_train.append(x_train[y])
    if(y_train[y]==8):
        new_y_train.append(1)
        new_x_train.append(x_train[y])
x_train = np.array(new_x_train)
y_train = np.array(new_y_train)
```
Finally, we perform an 80-20 train-validation split and store all of the datasets in a list.
```
trainIn, valIn, trainOut, valOut = train_test_split(
            x_train, y_train, test_size=0.20, random_state=21)
data=[trainIn, trainOut, valIn, valOut]
```

## Pretraining
Next, we will use the pretraining feature built into `TensorBNN`. This feature allows the use of normal neuarl network optimization algorithms to give us a superior starting point for the BNN as it is a much slower algorithm. Pretraining the networks then allows for faster convergence of the BNN. To do the pretraining, we simply call `trainBasicClassification` from the `BNNfunctions` folder. A sample call is shown below with all of the arguments labeled.
```
weights, biases, activation = trainBasicClassification(
      2, # Number of hidden layers
      inputDims, # Input dimensions
      outputDims, # Output dimensions
      20, # Number of perceptrons per layer 
      nncycles, # Number of training cylces. The learning rate is decreased by a factor of 10 each cycle.
      10, # Number of epochs per training cycle
      0.1, # Slope value for `leaky-relu` activation
      data[0], # Training input data
      data[1], # Training output data
      data[2], # Validation input data
      data[3], # Validation output data
      "MNIST_pretrain", # Save the pretrain network under this name
      callbacks=True, # Use callbacks to restore best weights obtained while training
      callbackMetric="val_loss", # metric used to determine best weights
      patience=10) # number of epochs to wait after failing to improve callback metric
```
Running this function will train a network in Keras and save it under the name "MNIST_pretrain". It will extract the weights, biases, and activation functions tensors from the final model and return them.

## BNN setup
We are now finally ready to actually setup the BNN.
First, we create a network object. This is accomplished through the following code:
```
dtype = tf.float32 # This is the best trade off between speed and precision.

neuralNet = network(
            dtype, # network datatype
            inputDims, # dimension of input vector
            data[0], # Training input data
            data[1], # Training output data
            data[2], # Validation input data
            data[3], # Validation output data
            tf.cast(0.0, dtype), # Mean of output data for unnormalization
            tf.cast(1.0, dtype)) # Standard deviation of output data
```
Next, we need to add our layers. We will use two hidden layers of 20 percpetrons each with SquarePrelu activation functions for the first two layers and Sigmoid activation for the output layers. SquarePrelu activations are similar to normal prelu activations, which are essentially leaky-relu activations with a trainable slope paramter. The difference, though, is that the trained parameter is the plus or minus square root of the slope. This way, we can not have an activation function which is not a bijection.
The code to add the layers is below:
```
seed = 0 # seed for layer generation, irrelavent with pretraining
width = 20 # number of perceptrons per layer
alpha = 0.1 # starting slope value for SquarePrelu
hidden = 2 # Number of hidden layers
neuralNet.add( # add a layer
    DenseLayer( # dense layer object
        inputDims, # input dimension
        width, # number of perceptrons per layer
        weights=weights[0], # pretrained weights
        biases=biases[0], # pretrained biases
        seed=seed, # layer seed
        dtype=dtype)) # layer datatype
neuralNet.add(SquarePrelu(width, 
                          alpha=alpha**(0.5), # starting slope parameter
                          activation=None, # no activation pretrained
                          dtype=dtype)) # activation datatype
seed += 1000
for n in range(hidden - 1): # Add the hidden layers
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

#Add the output layer
neuralNet.add(DenseLayer(width,
                         outputDims,
                         weights=weights[-1],
                         biases=biases[-1],
                         seed=seed,
                         dtype=dtype))
neuralNet.add(Sigmoid()) # Sigmoid activation
```
Next, we must setup the Markov Chain Monte Carlo algorithm. This is done by simply calling setupMCMC and providing it a lot of information.
```
neuralNet.setupMCMC(
        0.001, # Starting stepsize for Hamiltonian Monte Carlo (HMC)
        0.0005, # Minimum possible stepsize for HMC
        0.002, # Maximum possible stepsize for HMC
        100, # Number of points to use in stepsize search grid
        500, # Starting number of leapfrog steps for HMC
        100, # Minimum number of leapfrog steps for HMC
        2000, # Maximum number of leapfrog steps for HMC
        1, # increment in leapfrog steps in leapfrog search grid
        0.00001, # stepsize for hyper parameter HMC
        30, # leapfrog steps for hyper paramater HMC
        50, # Number of burnin steps to do
        2, # Number of cores to use on computer
        2) # Numberof stpes to average over in adaptive HMC algorithm
```
Finally, we can actually train the network. We must give it a few last pieces of information and then it will be on its merry way. 
```
neuralNet.train(
        2500, #Train for 2500 epochs
        10, # Save every 10 networks
        folderName="MNIST_BNN", # Save inside the folder MNIST_BNN
        networksPerFile=25, # Start new files every 25 networks
        returnPredictions=False, # Don't return predictions
        regression=False) # Don't use regression algorithm, so use classification algorithm 
```
A final word of caution: this algorithm is not fast. For large datasets and large networks it is only feasible to run this on GPUs, and even then it may need several days to run. This example is small enough that it should run on normal computers, but it will still take several hours. 
