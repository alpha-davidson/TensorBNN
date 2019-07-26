# TensorBNN
This package contains code which can be used to create full Bayesian Neural Networks using Hamiltonian Monte Carlo sampling as proposed by Radford Neal in his thesis "Bayesian Learning for Neural Networks" along with some added features. The package is written in python and uses the packages `Tensorflow` and `Tensorflow-Probability` as the framework for the implementation. 

## Dependencies
All python code written here is intended to be used in Python3. The code is dependent upon the packages numpy, tensorflow, tensorflow-probability, and scipy.

Numpy and scipy can be installed through the command:
```
pip3 install numpy scipy
```

The tensorflow version must be 2.0. Using a 1.x version will not work. It is also highly recomended that this code be run on a gpu due to its high computational complexit. Tensorflow 2.0 for the gpu can be installed with the command:
```
pip3 install tensorflow-gpu==2.0.0-beta1
```

In order to be compatible with tensorflow 2.0, the nightly version of tensorflow-probability must be installed. This is done with the following command:
'''
pip3 install tfp-nightly
'''

## Usage
Through the use of this package it is possible to easily make Bayesian Neural Networks for regression and binary classification learning problems. The folder `Examples` contains an excellent example of a regression problem and a binary classification problem. More generally, in order to use this code you must import network, Dense Layer, and an activation such as Relu. This can be done as follows:
```
from layer import DenseLayer
from network import network
from activationFunctions import Relu
```
Next, it is highly convenient to turn off the deprecation warnings. These are all from tensorflow, tensorflow-probability, and numpy intereacting with tensorflow, so it isn't something easily fixed and there are a lot of warnings. These are turned off with:
```
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```
The other important setup task is determining whether or not to seed the random number generator before training. Please note that if you are using a gpu then there will always be some randomness which cannot be removed. To set all cpu random numbers use these lines of code:
```
import os

import numpy as np
import random as rn
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)
```
Moving on to the actual use of this code, start with the declaration of a network object:
```
neuralNet = network.network(dtype, inputDims, trainX, trainY, validationX, validationY, mean, sd)
```
The paramaters are described as follows:
* dtype: data type for Tensors
* inputDims: dimension of input vector
* trainX: the training data input, shape is n by inputDims
* trainY: the training data output
* validateX: the validation data input, shape is n by inputDims
* validateY: the validation data output
* mean: the mean used to scale trainY and validateY
* sd: standard deviation used to scale trainY and validateY

Next, add all of the desired layers and activation functions as follows:
```
neuralNet.add(DenseLayer(inputDims, outputDims, seed=seed, dtype=tf.float32))
neuralNet.add(Relu())
```
For added control, especially when using pre-trained networks it is possible to feed pretrained weights, biases, and values for the activation functions. This can be done as follows:
```
neuralNet.add(DenseLayer(inputDims,outputDims, weights=weights, biases=biases, seed=seed, dtype=dtype))
neuralNet.add(SquarePrelu(width, alpha=alpha**(0.5), activation=activation, dtype=dtype))
```

The paramater inputDims is the output shape of the layer before, and the width is the ouput shape of the layers itself. The seed is used for seeding the random number generator. Currently, only ReLU is supported for easy predictions off of saved networks. The other activation functions can be used, but they will require more custom code to predict from saved networks.

Next, the Markov Chain Monte Carlo algorithm must be initialized. This can be done as follows:
```
neuralNet.setupMCMC(self, stepSize, stepMin, stepMax, stepNum, leapfrog, leapMin,
                    leapMax, leapStep, hyperStepSize, hyperLeapfrog, burnin,
                    cores, averagingSteps=2, a=4, delta=0.1):
```

The paramaters are described as follows:
* stepSize: the starting step size for the weights and biases
* stepMin: the minimum step size
* stepMax: the maximum step size
* stepNum: the number of step sizes in grid
* leapfrog: number of leapfrog steps for weights and biases
* leapMin: the minimum number of leapfrog steps
* leapMax: the maximum number of leapfrog steps
* leapStep: the step in number of leapfrog for search grid
* hyperStepSize: the starting step size for the hyper parameters
* hyperLeapfrog: leapfrog steps for hyper parameters
* cores: number of cores to use
* averaginSteps: number of averaging steps
* a: constant, 4 in paper
* delta: constant, 0.1 in paper

This code uses the adaptive Hamlitonain Monte Carlo described in "Adaptive Hamiltonian and Riemann Manifold Monte Carlo Samplers" by Wang, Mohamed, and de Freitas. In accordance with this paper there are a few more paramaters that can be adjusted, though it is recomended that their default values are kept.

The last thing to do is actually tell the model to start learning this is done with the following command:

```
network.train(epochs, startSampling, samplingStep, scaleExp=False, folderName=None, 
              networksPerFile=1000, returnPredictions=False, regression=True):
```
The arguments have the following meanings:

* Epochs: Number of training cycles
* startSampling: Number of epochs before networks start being saved
* samplingStep: Epochs between sampled networks
* scaleExp: whether the metrics should be scaled via exp
* folderName: name of folder for saved networks
* networksPerFile: number of networks saved in a given file
* returnPredictions: whether to return the prediction from the
                     network
* regression: for regression loss and metrics use True, for
              binary classification use False

Once the network has trained, which may take a while, the saved networks can be loaded and then used to make predictions using the following code:
```
import os

from BNN_functions import normalizeData, loadNetworks, predict

numNetworks, numMatrices, matrices=loadNetworks(filePath)

initialResults = predict(inputData, numNetworks, numMatrices, matrices)
```
The variable filePath is the directory from which the networks are being loaded, and inputData is the data for which predictions should be made.
