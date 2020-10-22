# TensorBNN
This package contains code which can be used to train Bayesian Neural Networks using Hamiltonian Monte Carlo sampling as proposed by Radford Neal in his thesis "Bayesian Learning for Neural Networks" along with added features. The package is written in python3 and uses the packages `Tensorflow` and `Tensorflow-Probability` as the framework for the implementation. 

For detailed information about this implementation, please see our paper on the arXiv: [TensorBNN: Bayesian Inference for Neural Networks using Tensorflow](https://arxiv.org/abs/2009.14393)

## Dependencies
All python code written here is in python3. The code is dependent upon the packages `numpy`, `tensorflow`, `tensorflow-probability`, and `scipy`.

The package, along with `numpy` and `scipy`, can be installed via

```
pip install tensorBNN
```

Alternatively, you can download `numpy` and `scipy` from source through the command:

```
pip3 install numpy scipy
```
TensorFlow and TensorFlow-probability must be instaled separately. The TensorFlow version should be the most recent (2.3 at the moment). Using a 1.x version will not work, and neither will older versions of 2. It is also highly recomended that this code be run on a gpu due to its high computational complexity. TensorFlow for the gpu can be installed with the command:

```
pip3 install tensorflow-gpu
```

In order to be compatible with this version of tensorflow, the most recent version of tensorflow-probability (0.11) must be installed. This is done with the following command:

```
pip3 install tensorflow-probability
```


## Usage

In order to use this code you must import network, Dense Layer, and an activation such as Relu. This can be done as follows:

```
from TensorBNN.layer import DenseLayer
from TensorBNN.network import network
from TensorBNN.activationFunctions import Relu
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

After initializaing the HMC, we must declare the likelihood that we want to use as well as any metrics. This can be accomplished through the following code:

```
# Declare Gaussian Likelihood with sd of 0.1
likelihood =  GaussianLikelihood(sd = 0.1)
metricList = [ #Declare metrics
    SquaredError(mean = 0, sd = 1, scaleExp = False),
    PercentError(mean = 10, sd = 2, scaleExp = True)]
```


The last thing to do is actually tell the model to start learning this is done with the following command:

```
network.train(
        epochs, # epochs to train for
        samplingStep, # increment between network saves
        likelihood,
        metricList = metricList,
        folderName = "Regression", 
        # Name of folder for saved networks
        networksPerFile=50)
        # Number of networks saved per file
```

The arguments have the following meanings:

* Epochs: Number of training cycles
* samplingStep: Epochs between sampled networks
* likelihood: The likelihood function used to evaluate the prediction 
              we defined previously
* startSigma: Starting standard deviation for likelihood function
              for regression models
* folderName: name of folder for saved networks
* networksPerFile: number of networks saved in a given file

Once the network has trained, which may take a while, the saved networks can be loaded and then used to make predictions using the following code:

```
from TensorBNN.predictor import predictor 

network = predictor(filePath,
                    dtype = dtype, 
                    # data type used by network
                    customLayerDict={"dense2": Dense2},
                    # A dense layer with a different 
                    # hyperprior
                    likelihood = Likelihood)
                    # The likelihood function is required to  
                    # calculate the probabilities for 
                    # re-weighting

initialResults = network.predict(inputData, skip, dtype)
```

The variable filePath is the directory from which the networks are being loaded, inputData is the normalized data for which predictions should be made, and dtype is the data type to be used for predictions. The customLayerDict is a dictionary holding the names and objects for any user defined layers. Likelihood is the likelihood function used to train the model.

The variable initialResults will be a list of numpy arrays, each numpy array corresponding to the predcitions from a single network in the BNN. The skip variable instructs the predictor to only use every n networks, where n=skip

Additionally, the predictor function allows for the calculation of the autocorrelation between different networks, as well as the autocorrelation length through:

```
autocorrelations = network.autocorrelation(testData, nMax)
autocorrelations = network.autoCorrelationLength(testData, nMax)
```
Here, the autocorrelation is calculated based on the predictions of the different BNNs, and the results are averaged over the test data. nMax provides the largest lag value for the autocorrelation. These calculations are done with emcee.


Finally, the predictor object can calculate new weights for the different networks if they were given new priors. These priors take the form of new Layer objects which must be referenced in an architecture file. The reweighting function call looks like this:

```
weights = network.reweight(                                            
                    trainX, # training input
                    trainY, # training output
                    skip = 10, # Use every 10 saved networks
                    architecture = "architecture2.txt")
                    # New architecture file
```

