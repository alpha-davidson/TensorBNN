import tensorflow as tf
import tensorflow_probability as tfp

from tensorBNN.BNN_functions import multivariateLogProb

tfd = tfp.distributions

class Likelihood(object):
    def __init__(self, *argv, **kwargs):
        """
        When declared, this constructor will be given keywords corresponding
        to any possible hyper parameters, as well as whether it should be
        calculated when the hyperparameters are being adjusted. This will
        likely be the case if it has hyper paramters.
        """
        self.hypers=[]
        self.mainProbsInHypers=False
    
    def makeResponseLikelihood(self, *argv, **kwargs):
        """ This method will make a prediction and predict its probability
        given the likelihood funtion implemented. It will need at least the
        following inputs and must have the following outputs:
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * predict: the function used to make a prediction from the 
            current neural net
            * dtype: the datatype of the network
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values 
        """
        self.hypers=[]
        
    def calculateLogProb(self, *argv, **kwargs):
        """ This is a version of makeResponseLikelihood designed to deal with
        multiple sets of hyper paramters. It is used for reweighting in the
        predictor object, not during training as makeResponseLikelihood is.
        It also requires at least the following inputs and outputs:    
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * hypers: A list containing all the hyper paramters
            * predict: the function used to make a prediction from the 
            current neural net
            * dtype: the datatype of the network
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values 
        
        """
        pass
    
    def display(self, hypers):
        """An optional method which can be used to display relavent information
        during the evaluation phase of a network.
        """
        pass
    

class GaussianLikelihood(Likelihood):
    
    def __init__(self, *argv, **kwargs):
        self.hypers=[kwargs["sd"]]
        self.mainProbsInHypers = True
    
    @tf.function
    def makeResponseLikelihood(self, *argv, **kwargs):
        """Make a prediction and predict its probability from a multivariate
        normal distribution
    
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * sd: standard deviation for output distribution, uses
            current hyper parameter value if nothing is given
            * hyperStates: A list containing all the hyper paramters
            * predict: the function used to make a prediction from the 
            current neural net
            * dtype: the datatype of the network
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values
        """
    
        if(kwargs["sd"] is None):
            sd = kwargs["hyperStates"][-1]
        else:
            sd = kwargs["sd"]
        current = kwargs["predict"](True, argv[0])
        current = tf.transpose(current)
        sigma = tf.ones_like(current) * sd
        realVals = tf.reshape(kwargs["realVals"], current.shape)
        result = multivariateLogProb(sigma, current, realVals, kwargs["dtype"])
    
        return(result)
    
    def calcultateLogProb(self, *argv, **kwargs):
        """Make a prediction and predict its probability from a multivariate
        normal distribution
    
        rguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * hypers: A list containing all the hyper paramters
            * predict: the function used to make a prediction from the 
            current neural net
            * dtype: the datatype of the network
            * skip: Use every n networks where n=skip
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values
        """
        sd = []
        for x in range(len(kwargs["hypers"])):
            sd.append(kwargs["hypers"][x][-1])

        current = kwargs["predict"](argv[0], skip=kwargs["skip"])
        for x in range(len(current)):
            current[x] = tf.transpose(current[x])
        
        realVals = tf.reshape(kwargs["realVals"], current[0].shape)
        result = []
        for x in range(len(current)):
            result.append(multivariateLogProb(tf.ones_like(current[0]) * sd[x],
                                              current[x], realVals,
                                              kwargs["dtype"]))
    
        return(result)
    
    def display(self, hypers):
        print("Loss Standard Deviation: ", hypers[-1].numpy())


class BernoulliLikelihood(Likelihood):
    def __init__(self, *argv, **kwargs):
        self.hypers=[]
        self.mainProbsInHypers=False
    
    @tf.function
    def makeResponseLikelihood(self, *argv,  **kwargs):
        """Make a prediction and predict its probability from a Bernoulli
           normal distribution
    
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * predict: the function used to make a prediction from the 
            current neural net
            * dtype: the datatype of the network
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values
        """
        current = kwargs["predict"](True, argv[0])
        current = tf.cast(
            tf.clip_by_value(
                current,
                1e-8,
                1 - 1e-7),
            kwargs["dtype"])
        
        # Prediction distribution
        dist = tfd.Bernoulli(
            probs=current)
        result = dist.log_prob(tf.transpose(kwargs["realVals"]))
        return(result)
        
    def calcultateLogProb(self, *argv, **kwargs):
        result=[]
        for x in range(len(kwargs["hypers"])):
            result.append(tf.cast(0,kwargs["dtype"]))
        return(result)