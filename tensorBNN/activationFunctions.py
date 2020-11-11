import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import gen_nn_ops

from tensorBNN.BNN_functions import multivariateLogProb
from tensorBNN.layer import Layer

tfd = tfp.distributions


class Relu(Layer):
    """Relu activation function"""

    def __init__(self, inputDims=None, outputDims=None):
        self.numTensors = 0
        self.numHyperTensors = 0
        self.name = "relu"

    def predict(self, inputTensor, _):
        result = gen_nn_ops.relu(inputTensor)
        return(result)


class Sigmoid(Layer):
    """Sigmoid activation function"""

    def __init__(self, inputDims=None, outputDims=None):
        self.numTensors = 0
        self.numHyperTensors = 0
        self.name = "sigmoid"

    def predict(self, inputTensor, _):
        result = tf.math.sigmoid(inputTensor)
        return(result)


class Tanh(Layer):
    """Tanh activation function"""

    def __init__(self, inputDims=None, outputDims=None):
        self.numTensors = 0
        self.numHyperTensors = 0
        self.name = "tanh"

    def predict(self, inputTensor, _):
        result = tf.math.tanh(inputTensor)
        return(result)


class Elu(Layer):
    """Elu activation function"""

    def __init__(self, inputDims=None, outputDims=None):
        self.numTensors = 0
        self.numHyperTensors = 0
        self.name = "elu"

    def predict(self, inputTensor, _):
        result = gen_nn_ops.elu(inputTensor)
        return(result)


class Softmax(Layer):
    """Softmax activation function"""

    def __init__(self, inputDims=None, outputDims=None):
        self.numTensors = 0
        self.numHyperTensors = 0
        self.name = "softmax"

    def predict(self, inputTensor, _):
        result = gen_nn_ops.softmax(inputTensor)
        return(result)


class Leaky_relu(Layer):
    """Leaky relu activation function"""

    def __init__(self, alpha=0.3, inputDims=None, outputDims=None,
                 activation=None):
        self.numTensors = 1
        self.numHyperTensors = 0
        self.name = "leakyrelu"
        if activation is not None:
            alpha = activation
        self.parameters = [alpha]

    def predict(self, inputTensor, _):
        result = tf.nn.leaky_relu(inputTensor, self.parameters[0])
        return(result)

    def calculateProbs(self, *args):
        """Present for compatability."""
        return(0.0)

    def updateParameters(self, *args):
        """Present for compatability."""
        self.parameters = self.parameters


class Prelu(Layer):
    """Prelu activation function"""

    def __init__(
            self,
            inputDims,
            outputDims=None,
            dtype=np.float32,
            alpha=0.2,
            activation=None,
            seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * dtype: data type of input and output values
            * alpha: Single custom starting slope value
            * activation: optional custom values for starting slope values
            * seed: seed used for random numbers
        """
        self.numTensors = 1  # Number of tensors used for predictions
        self.numHyperTensors = 1  # Number of tensor for hyper paramaters
        self.inputDims = inputDims
        self.dtype = dtype
        self.seed = seed
        self.name = "prelu"

        # Starting rate value and hyperRate
        rate = tf.cast(0.3, dtype)
        self.hyperRate = tf.cast(0.3, self.dtype)

        # Starting weight mean, weight SD, bias mean, and bias SD

        self.hypers = [tf.cast(rate, self.dtype)]

        # Starting weights and biases
        if(activation is None):
            self.parameters = [
                alpha *
                tf.ones(
                    shape=(inputDims),
                    dtype=self.dtype)]
        else:
            self.parameters = [activation]

    @tf.function
    def exponentialLogProb(self, rate, x):
        """Calcualtes the log probability of an exponential distribution.

        Arguments:
            * rate: rate parameter for the distribution
            * x: input value
        Returns:
            * logProb: log probability of x
        """

        rate = tf.math.abs(rate)
        logProb = -rate * x + tf.math.log(rate)

        return(logProb)

    @tf.function
    def calculateProbs(self, slopes):
        """Calculates the log probability of the slopes given
        their distributions in this layer.

        Arguments:
            * weightsBias: list with new possible weight and bias tensors

        Returns:
            * prob: log prob of weights and biases given their distributions
        """

        val = self.exponentialLogProb(self.hypers[0], slopes)
        prob = tf.reduce_sum(input_tensor=val)

        return(prob)

    @tf.function
    def calculateHyperProbs(self, hypers, slopes):
        """Calculates the log probability of a set of weights and biases given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.

        Arguments:
            * hypers: a list containg 4 new possible hyper parameters
            * weightBias: a list with the current weight and bias matrices

        Returns:
            * prob: log probability of weights and biases given the new hypers
            and the probability of the new hyper parameters given their priors
        """

        slopes = tf.math.abs(slopes[0])
        prob = 0

        # Calculate probability of new hypers
        val = self.exponentialLogProb(self.hyperRate, hypers[0])
        prob += tf.reduce_sum(input_tensor=val)

        # Calculate probability of weights and biases given new hypers
        val = self.exponentialLogProb(hypers[0], slopes)
        prob += tf.reduce_sum(input_tensor=val)

        return(prob)

    @tf.function
    def expand(self, current):
        """Expands tensors to that they are of rank 2

        Arguments:
            * current: tensor to expand
        Returns:
            * expanded: expanded tensor

        """
        currentShape = tf.pad(
            tensor=tf.shape(input=current),
            paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
            constant_values=1)
        expanded = tf.reshape(current, currentShape)
        return(expanded)

    def predict(self, inputTensor, slopes):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values

        Arguments:
            * inputTensor: the input tensor the layer acts on
            * weightBias: a list with the current weight and bias tensors
        Returns:
            * result: the output of the layer

        """
        slopes = slopes[0]
        slopes = tf.reshape(slopes, (len(slopes), 1))
        activated = tf.multiply(slopes, inputTensor)
        result = tf.where(tf.math.less(inputTensor, 0), activated, inputTensor)
        return(self.expand(result))

    def updateParameters(self, slopes):
        """ Updates the network parameters

        Arguments:
            * slopes: new slope parameter
        """
        self.parameters = [slopes[0]]

    def updateHypers(self, hypers):
        """ Updates the network parameters

        Arguments:
            * slopes: new slope parameter
        """
        self.hypers = [tf.maximum(tf.cast(0.01, self.dtype), hypers[0])]


class SquarePrelu(Layer):
    """Prelu activation function"""

    def __init__(
            self,
            inputDims,
            outputDims=None,
            dtype=np.float32,
            alpha=0.2,
            activation=None,
            seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * dtype: data type of input and output values
            * alpha: Single custom starting slope value
            * activation: optional custom values for starting slope values
            * seed: seed used for random numbers
        """
        self.numTensors = 1  # Number of tensors used for predictions
        self.numHyperTensors = 2  # Number of tensor for hyper paramaters
        self.inputDims = inputDims
        self.dtype = dtype
        self.seed = seed
        self.name = "squareprelu"

        # Starting rate value and hyperRate
        mean = tf.cast(0.0, dtype)
        sd = tf.cast(0.3, dtype)

        meanMean = tf.cast(0.0, dtype)
        meanSD = tf.cast(0.3, dtype)
        sdMean = tf.cast(0.3, dtype)
        sdSD = tf.cast(0.1, dtype)

        self.meanHyper = tfd.MultivariateNormalDiag(loc=[meanMean],
                                                    scale_diag=[meanSD])

        self.sdHyper = tfd.MultivariateNormalDiag(loc=[sdMean],
                                                  scale_diag=[sdSD])

        # Starting weight mean, weight SD, bias mean, and bias SD

        self.hypers = [mean, sd]

        # Starting weights and biases
        if(activation is None):
            self.parameters = [
                alpha *
                tf.ones(
                    shape=(inputDims),
                    dtype=self.dtype)]
        else:
            self.parameters = [activation]

    @tf.function
    def calculateProbs(self, slopes):
        """Calculates the log probability of the slopes given
        their distributions in this layer.

        Arguments:
            * weightsBias: list with new possible weight and bias tensors

        Returns:
            * prob: log prob of weights and biases given their distributions
        """

        prob = tf.reduce_sum(
            multivariateLogProb(
                self.hypers[1],
                self.hypers[0],
                slopes,
                dtype=self.dtype))

        return(prob)

    @tf.function
    def calculateHyperProbs(self, hypers, slopes):
        """Calculates the log probability of a set of weights and biases given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.

        Arguments:
            * hypers: a list containg 4 new possible hyper parameters
            * weightBias: a list with the current weight and bias matrices

        Returns:
            * prob: log probability of weights and biases given the new hypers
            and the probability of the new hyper parameters given their priors
        """

        mean = hypers[0]
        sd = hypers[1]

        slopes = tf.square(slopes[0])

        prob = tf.reduce_sum(
            multivariateLogProb(
                sd, mean, slopes, dtype=self.dtype))

        # Calculate probability of new hypers
        val = self.meanHyper.log_prob([mean])
        prob += tf.reduce_sum(input_tensor=val)

        # Calculate probability of weights and biases given new hypers
        val = self.sdHyper.log_prob([sd])
        prob += tf.reduce_sum(input_tensor=val)

        return(prob)

    @tf.function
    def expand(self, current):
        """Expands tensors to that they are of rank 2

        Arguments:
            * current: tensor to expand
        Returns:
            * expanded: expanded tensor

        """
        currentShape = tf.pad(
            tensor=tf.shape(input=current),
            paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
            constant_values=1)
        expanded = tf.reshape(current, currentShape)
        return(expanded)

    def predict(self, inputTensor, slopes):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values

        Arguments:
            * inputTensor: the input tensor the layer acts on
            * weightBias: a list with the current weight and bias tensors
        Returns:
            * result: the output of the layer

        """
        slopes = slopes[0]**2
        slopes = tf.reshape(slopes, (len(slopes), 1))
        activated = tf.multiply(slopes, inputTensor)
        result = tf.where(tf.math.less(inputTensor, 0), activated, inputTensor)
        return(self.expand(result))

    def updateParameters(self, slopes):
        """ Updates the network parameters

        Arguments:
            * slopes: new slope parameter
        """
        self.parameters = [slopes[0]]

    def updateHypers(self, hypers):
        """ Updates the network parameters

        Arguments:
            * slopes: new slope parameter
        """
        self.hypers = [hypers[0], hypers[1]]
