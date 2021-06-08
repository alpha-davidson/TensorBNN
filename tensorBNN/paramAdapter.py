import math
import random
import sys

import numpy as np
import tensorflow as tf

from multiprocessing import Pool


class paramAdapter(object):
    """This object stores the variables required to implement an adaptive
    step size and number of leapfrog steps as detailed in "Adaptive Hamiltonian
    and Riemann Manifold Monte Carlo Samplers" by Wang, Mohamed, and
    de Freitas. This method performs Bayesian inference on these paramaters
    assuming a uniform prior between specified values. Over time, the
    probability of a new state being proposed decreases so that the values will
    converge to specific values.

    In a slight divergence from the paper three features are included to
    prevent the adapter from settling to early into an non-optimal position, to
    compensate for the optimal position chaning drastically through trainining,
    and to generally improve the suggested points. First, the adapter will
    randomly propose for a certain number of steps at the beginning as set by
    the randomSteps keyword argument. Secondly, if the adapter goes through a
    set number of iterations specified with the strikes keyword argument and
    the SJD is 0 every single time then the entire paramAdapter is reset to its
    initial condition. It is quite possible that this will happen after the BNN
    converges to a minimum and the maximum feasible step size is much smaller.
    Finally, the adapter will scale the leapfrog steps and step size to the
    range -1 to 1 in order for the 0 mean Gaussian priors used in the Bayeisan
    inference to better fit the data.

    In order to more rapidly search through the grid of possible step sizes
    and leapfrog steps this object uses parallel processing so that all
    available computing resources are used.
    """

    def __init__(self, e1, L1, el, eu, eNumber, Ll, Lu, lStep, m, k, a=4,
                 delta=0.1, cores=4, strikes=10, randomSteps=10):
        """ Creates a paramAdapter object.

        Arguments:
            * e1: starting step size
            * L1: starting number of leapfrog steps
            * el: lower step size bound
            * eu: upper step size bound
            * eNumber: number of step sizes in gride
            * Ll: lower leapfrog bound
            * Lu: upper leapfrog bound
            * lStep: leapfrog step size in grid
            * m: number of averaging steps
            * k: iterations before proposal probability starts decreasing
            * a: constant, 4 in paper
            * delta: constant, 0.1 in paper
            * cores: number of cores to use in processing
            * strikes: iterations with no movement before reseting adapter
            * randomSteps: averaging cycles at beginning with random values
        """
        self.dtype=tf.float32
        self.currentE = e1
        self.currentL = L1
        self.el = tf.cast(el, self.dtype)
        self.eu = tf.cast(eu, self.dtype)
        self.Ll = tf.cast(Ll, self.dtype)
        self.Lu = tf.cast(Lu, self.dtype)
        self.eNumber = tf.cast(eNumber, tf.int32)
        self.eGrid = tf.linspace(el, eu, num=eNumber)
        self.lGrid = tf.cast(np.array(range(Ll, Lu + 1, int(lStep))), self.dtype)
        self.lNumber = tf.cast(len(self.lGrid), tf.int32)
        self.delta = tf.cast(delta, self.dtype)
        kappa = tf.cast(0.2, self.dtype)
        self.sigma = tf.linalg.diag(
            [1 / ((kappa * (2))**2), 1 / ((kappa * (2))**2)])
        self.previousGamma = []

        self.allSD = []
        self.k = k
        self.K = tf.zeros([0,0], dtype=self.dtype)
        self.m = m
        self.currentData = []
        self.allData = []
        self.maxR = tf.cast(1e-8, self.dtype)
        self.a = tf.cast(a, self.dtype)
        self.i = tf.cast(-2, self.dtype)
        self.previous_state = None
        self.current_state = None
        #np.random.seed(10)

        self.cores = cores
        self.strikes = 0
        self.maxStrikes = 50#strikes
        self.randomSteps = randomSteps

    def calck(self, gammaI, gammaJ, el, eu, sigma):
        """ Calculates the covariance k between two states

        Arguments:
            * gammaI: state 1
            * gammaJ: state 2
        Returns:
            * k: covaraiance between gammaI and gammaJ
        """
        gamma1 = tf.transpose([[-1+2*(gammaI[0]-el)/(eu-el),
                  -1+2*(tf.cast(gammaI[1], self.dtype)-self.Ll)/(self.Lu-self.Ll)]])
        gamma2 = tf.transpose([[-1+2*(gammaJ[0]-el)/(eu-el),
                  -1+2*(tf.cast(gammaJ[1], self.dtype)-self.Ll)/(self.Lu-self.Ll)]])

        k = tf.exp(-0.5 * (tf.matmul(tf.transpose(gamma1),
                                     tf.matmul(sigma, gamma2))))
        return(k)

    def calcUCB(self, testGamma, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma):
        """ Calculates a varraint of the upper confidence bound for a test
        state.

        Arguments:
            * testGamma: the test state
            * s: a scaling factor
            * inverse: inverse of the covariance matrix
            * inverseR: inverse of the covariance matrix time the data
            * p: the decay value
            * rootBeta: a constant based on the number of variables in the
                        state
        Returns:
            * ucb: upper confidence bound
        """
        k = []#[None] * self.inverse.shape[0]
        for gamma, index in zip(previousGamma,
                                range(len(previousGamma))):
            #k[index] = self.calck(gamma, testGamma)
            k.append([self.calck(gamma, testGamma, el, eu, sigma)[0,0]])
        k = tf.cast(k, self.dtype)
        mean = tf.matmul(tf.transpose(k), inverseR) * s
        variance = tf.matmul(inverse, k)
        variance = tf.matmul(tf.transpose(k), variance)
        
        variance = self.calck(testGamma, testGamma, el, eu, sigma) - variance
        
        ucb = mean + variance * p * rootbeta
        return(ucb, mean, variance)

    def reset(self):
        """Resets the adapter"""
        tf.print("Reset")
        self.previousGamma = []

        self.allSD = []
        self.K = tf.zeros([0,0])
        self.currentData = []
        self.allData = []
        self.maxR = 1e-8
        self.i = -2
        self.previous_state = None
        self.current_state = None
        self.strikes = 0

    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def gridSearch(self, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma):
        eCount = tf.constant(0, dtype=tf.int32)
        lCount = tf.constant(0, dtype=tf.int32)
        cond = lambda eCount, lCount, e, L, ucb, mean, variance, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma: tf.less(lCount, self.lNumber)
        e = tf.cast([[el]], self.dtype)
        L = tf.cast([[self.Ll]], self.dtype)
        ucb = tf.cast([[-1000000000]], self.dtype)
        variance = tf.cast([[-1000000000]], self.dtype)
        mean=tf.cast([[-1000000000]], self.dtype)
        
        
        def processChunk(eCount, lCount, e, L, ucb, mean, variance, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma):
            """Processes a chunk of the e, L combinations.
    
            Arguments:
                * eList: list of step sizes to check
                * lList: list of leapfrog steps to check
    
            Returns:
                * best: a tuple of the form ((best e, best L), ucb) where the e and
                L selected are those with the highest ucb, which is also included
            """
            newE = self.eGrid[eCount]
            newL = self.lGrid[lCount]
            newUcb, newMean, newVariance = self.calcUCB([newE, newL], previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma)
            e = tf.where(newUcb>ucb,newE, e)
            L = tf.where(newUcb>ucb,newL, L)
            mean = tf.where(newUcb>ucb,newMean, mean)
            variance = tf.where(newUcb>ucb,newVariance, variance)
            ucb = tf.where(newUcb>ucb,newUcb, ucb)
            lCount = tf.where(eCount==self.eNumber-1, lCount+1, lCount)
            eCount = tf.where(eCount==self.eNumber-1, 0, eCount+1)
            return(eCount, lCount, e, L, ucb, mean, variance, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma)
        
        eCount, lCount, e, L, ucb, mean, variance, previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma = tf.while_loop(cond, processChunk, [eCount, lCount, e, L, ucb, mean, variance,
                                                                                       previousGamma, inverseR, s, inverse, p, rootbeta, el, eu, sigma])
        
        return(tf.cast(e[0,0], self.dtype), tf.cast(L[0,0], self.dtype))


    def update(self, state):
        """ Steps the adapter forward by one step

        Arguments:
            * state: the newest state proposed by the HMC algorithm
        Returns:
            * currentE: the new step size
            * currentL: the new number of leapfrog steps
        """
        if(self.i<self.k-2 and self.strikes == self.maxStrikes):
            self.el = self.el/2
            self.eu = self.eu/2
            self.eGrid =tf.linspace(self.el, self.eu, num=self.eNumber)
            self.k=self.k-self.i-2
            self.reset()
            self.strikes = 0

        self.previous_state, self.current_state = self.current_state, state

        # Calculate the square jumping distance scaled by L^(-0.5)
        if(self.previous_state is not None):
            val = tf.cast(0, tf.float32)
            for old, new in zip(self.previous_state, self.current_state):
                val += tf.math.reduce_sum(tf.math.square(tf.reshape(new,[-1]) - tf.reshape(old,[-1]))) / (tf.cast(self.currentL, tf.float32))**(0.5)
            print("SJD:", val.numpy())
            self.currentData.append(val)
            if(val < 1e-8 and self.i // self.m > self.randomSteps):
                self.strikes += 1
            else:
                self.strikes = 0

        # Update E and L if this is not just an averaging step
        if(self.i % self.m == 0 and self.i > 0):
            u = tf.random.uniform([1,1],minval=0, maxval=1)
            self.p = max(self.i / self.m - self.k + 1, 1)**(-0.5)
            if(u < self.p+u*0):  # Over time the probability of updating will decay
                mean = tf.math.reduce_mean(self.currentData)
                sd = tf.math.reduce_std(self.currentData)
                self.currentData = []
                self.allData.append(mean)
                self.allSD.append(sd)
                self.maxR = tf.math.reduce_max(self.allData)
                # Update the covariance matrix
                self.previousGamma.append((self.currentE, self.currentL))
                size = len(self.previousGamma)
                newK = tf.ones([size, size])
                if(size > 0):
                    #newK[:size - 1, :size - 1] = self.K
                    newK = self.K
                newKExtra=[]
                for gamma, index in zip(self.previousGamma, range(
                        len(self.previousGamma))):
                    k = self.calck(gamma, self.previousGamma[-1], self.el, self.eu, self.sigma)
                    #newK[-1, index] = k
                    #newK[index, -1] = k
                    newKExtra.append(k[0,0])
                newK = tf.concat([newK, [newKExtra[:-1]]],axis=0)
                newK = tf.concat([newK, tf.transpose([newKExtra])], axis=1)
                self.K = newK
                self.s = self.a / self.maxR  # update scalling constant

                sigmaNu = tf.math.reduce_mean(self.allSD)  # Variance of noise

                # calculate inverse and other values only once
                try:  # In case the covaraince matrix is singular
                    self.inverse = tf.linalg.inv(
                        self.K + (sigmaNu**2) * tf.eye(self.K.shape[0]))
                except tf.errors.InvalidArgumentError:
                    self.inverse = tf.linalg.inv(
                        self.K + (sigmaNu**2) * tf.eye(self.K.shape[0]) +
                        0.1 * tf.eye(self.K.shape[0]))
                self.inverseR = tf.matmul(self.inverse, tf.expand_dims(tf.cast(self.allData, tf.float32),1))
                
                
                
                self.rootbeta = (self.i / self.m + 1)**(3) * math.pi**2
                self.rootbeta /= (3 * self.delta)
                self.rootbeta = tf.math.log(self.rootbeta)*2
                self.rootbeta = self.rootbeta**(0.5)

                # Start parallel searches, take best result found
                if(self.i//self.m >= self.randomSteps):
                    self.currentE, self.currentL = self.gridSearch(self.previousGamma, self.inverseR, self.s, self.inverse, self.p, self.rootbeta, self.el, self.eu, self.sigma)
                else:
                    self.currentE = random.choice(self.eGrid)
                    self.currentL = random.choice(self.lGrid)
                if(size==50):
                    self.K=self.K[1:,1:]
                    self.previousGamma=self.previousGamma[1:]
                    self.allData=self.allData[1:]
                    self.allSD=self.allSD[1:]

        self.i += 1
        return(tf.cast(self.currentE, self.dtype), tf.cast(self.currentL, tf.int32))
