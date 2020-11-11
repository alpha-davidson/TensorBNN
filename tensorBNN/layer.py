import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorBNN.BNN_functions import cauchyLogProb

tfd = tfp.distributions


class Layer(object):
    """ A basic layer object. This must have input and output dimensions, but
    the remaining variables can be used depending on the specific use. This
    object can be implemented as a basic layer or as an activation function.
    """

    def __init__(
            self,
            inputDims,
            outputDims,
            weights=None,
            biases=None,
            activation=None,
            dtype=np.float32,
            alpha=0,
            seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * outputDims: number of output dimensions
            * weights: list of starting weight matrices
            * biases: list of starting bias vectors
            * activation: list of starting activation function values
            * dtype: data type of input and output values
            * alpha: constant used for activation functions
            * seed: seed used for random numbers
        """
        self.numTensors = 0  # Number of tensors used for predictions
        self.numHyperTensors = 0  # Number of tensor for hyper paramaters
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.dtype = dtype
        self.seed = seed
        self.name = "name"


    def calculateProbs(self, tensors):
        """Calculates the log probability of a set of tensors given
        their distributions in this layer.

        Arguments:
            * tensors: list with new possible tensors for layer

        Returns:
            * prob: log prob of new tensors given their distributions
        """
        return(tf.Constant(0.0, shape=(), dtype=tf.float32))
       
    def calculateHyperProbs(self, hypers, tensors):
        """Calculates the log probability of a set of tensors given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.

        Arguments:
            * hypers: a list containg new possible hyper parameters
            * tensors: a list with the current tensors

        Returns:
            * prob: log probability of tensors given the new hypers
            and the probability of the new hyper parameters given their priors
        """
        return(tf.constant(0.0))

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

    def predict(self, inputTensor, tensors):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values

        Arguments:
            * inputTensor: the input tensor the layer acts on
            * tensors: a list with the current layer tensors
        Returns:
            * result: the output of the layer
        """
        pass

    def updateParameters(self, tensors):
        self.parameters = tensors

    def updateHypers(self, hypers):
        self.hypers = hypers

class DenseLayer(Layer):
    """Creates a 1 Dimensional Dense Bayesian Layer.

    Currently, the starting weight and bias mean values are 0.0 with a standard
    deviation of 1.0/sqrt(outputDims). The distribution that these values are
    subject to have these values as their means, and a standard deviation of
    2.0/sqrt(outputDims).
    """

    def __init__(
            self,
            inputDims,
            outputDims,
            weights=None,
            biases=None,
            dtype=np.float32,
            seed=1):
        """
        Arguments:
            * inputDims: number of input dimensions
            * outputDims: number of output dimensions
            * weights: list of starting weight matrices
            * biases: list of starting bias vectors
            * dtype: data type of input and output values
            * seed: seed used for random numbers
        """
        self.numTensors = 2  # Number of tensors used for predictions
        self.numHyperTensors = 4  # Number of tensor for hyper paramaters
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.dtype = dtype
        self.seed = seed
        self.name = "dense"

        # Weight mean value and mean distribution
        weightsx0 = 0.0
        self.weightsx0Hyper = tfd.MultivariateNormalDiag(loc=[weightsx0],
                                                         scale_diag=[.2])

        # weight SD value and SD distribution
        weightsGamma = 0.5
        self.weightsGammaHyper = tfd.MultivariateNormalDiag(loc=[weightsGamma],
                                                            scale_diag=[0.5])

        # bias mean value and mean distribution
        biasesx0 = 0.0
        self.biasesx0Hyper = tfd.MultivariateNormalDiag(loc=[biasesx0],
                                                        scale_diag=[.2])

        # bias SD value and SD distribution
        biasesGamma = 0.5
        self.biasesGammaHyper = tfd.MultivariateNormalDiag(loc=[biasesGamma],
                                                           scale_diag=[0.5])

        # Starting weight mean, weight SD, bias mean, and bias SD
        self.hypers = tf.cast(
            [weightsx0, weightsGamma, biasesx0, biasesGamma], self.dtype)

        # Starting weights and biases
        if(weights is None):
            self.parameters = self.sample()
        else:
            self.parameters = [weights, biases]
            
    def setHyperDistribtuions(self, weightMeanMean, weightMeanSD,
                              weightGammaMean, weightGammaSD,
                              biasMeanMean, biasMeanSD, biasGammaMean,
                              biasGammaSD):
        
        # Weight mean value and mean distribution
        self.weightsx0Hyper = tfd.MultivariateNormalDiag(loc=[weightMeanMean],
                                                         scale_diag =
                                                         [weightMeanSD])

        # weight SD value and SD distribution
        self.weightsGammaHyper = tfd.MultivariateNormalDiag(loc=[weightGammaMean],
                                                            scale_diag=[weightGammaSD])

        # bias mean value and mean distribution
        self.biasesx0Hyper = tfd.MultivariateNormalDiag(loc=[biasMeanMean],
                                                        scale_diag=[biasMeanSD])

        # bias SD value and SD distribution
        self.biasesGammaHyper = tfd.MultivariateNormalDiag(loc=[biasGammaMean],
                                                           scale_diag=[biasGammaSD])

    def calculateProbs(self, tensors):
        """Calculates the log probability of a set of weights and biases given
        their distributions in this layer.

        Arguments:
            * weightsBias: list with new possible weight and bias tensors

        Returns:
            * prob: log prob of weights and biases given their distributions
        """
        # Create the tensors used to calculate probability
        weights = tensors[0]
        biases = tensors[1]
        weightsx0 = self.hypers[0]
        weightsGamma = self.hypers[1]
        biasesx0 = self.hypers[2]
        biasesGamma = self.hypers[3]
        prob = 0

        # Calculate the probability of the paramaters given the current hypers
        val = cauchyLogProb(weightsGamma, weightsx0, weights, dtype=self.dtype)
        prob += tf.reduce_sum(input_tensor=val)
        val = cauchyLogProb(
            biasesGamma,
            biasesx0,
            biases,
            dtype=self.dtype)
        prob += tf.reduce_sum(input_tensor=val)
        return(prob)

    def calculateHyperProbs(self, hypers, tensors):
        """Calculates the log probability of a set of weights and biases given
        new distribtuions as well as the probability of the new distribution
        means and SDs given their distribtuions.

        Arguments:
            * hypers: a list containg 4 new possible hyper parameters
            * tensors: a list with the current weight and bias matrices

        Returns:
            * prob: log probability of weights and biases given the new hypers
            and the probability of the new hyper parameters given their priors
        """
        weightsx0 = hypers[0]
        weightsGamma = hypers[1]
        biasesx0 = hypers[2]
        biasesGamma = hypers[3]
        weights = tensors[0]
        biases = tensors[1]

       
        prob = tf.cast(0, self.dtype)
        
        val = self.weightsx0Hyper.log_prob([[weightsx0]])
        prob += tf.cast(tf.reduce_sum(input_tensor=val), self.dtype)
        val = self.weightsGammaHyper.log_prob([[weightsGamma]])
        prob += tf.cast(tf.reduce_sum(input_tensor=val), self.dtype)

        val = self.biasesx0Hyper.log_prob([[biasesx0]])
        prob += tf.cast(tf.reduce_sum(input_tensor=val), self.dtype)
        val = self.biasesGammaHyper.log_prob([[biasesGamma]])
        prob += tf.cast(tf.reduce_sum(input_tensor=val), self.dtype)
        
        # Calculate probability of weights and biases given new hypers
        val = cauchyLogProb(weightsGamma, weightsx0, weights, dtype=self.dtype)
        prob += tf.reduce_sum(input_tensor=val)
        val = cauchyLogProb(
            biasesGamma,
            biasesx0,
            biases,
            dtype=self.dtype)
        prob += tf.reduce_sum(input_tensor=val)

        return(prob)

    def sample(self):
        """Creates randomized weight and bias tensors based off
        of their distributions

        Returns:
            * tempWeights: randomized weight tensor in first list position
            * tempBiases: randomized bias tensor in second list position
        """

        tempWeights = tf.random.normal((self.outputDims, self.inputDims),
                                       mean=self.hypers[0],
                                       stddev=(2 / self.outputDims)**(0.5),
                                       seed=self.seed,
                                       dtype=self.dtype)
        tempBiases = tf.random.normal((self.outputDims, 1),
                                      mean=self.hypers[2],
                                      stddev=(2 / self.outputDims)**(0.5),
                                      seed=self.seed + 1,
                                      dtype=self.dtype)

        return([tempWeights, tempBiases])

    def predict(self, inputTensor, tensors):
        """Calculates the output of the layer based on the given input tensor
        and weight and bias values

        Arguments:
            * inputTensor: the input tensor the layer acts on
            * tensors: a list with the current weight and bias tensors
        Returns:
            * result: the output of the layer
        """

        weightTensor = self.expand(tensors[0])
        biasTensor = self.expand(tensors[1])
        result = tf.add(tf.matmul(weightTensor, inputTensor), biasTensor)
        return(result)

