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
        self.currentE = e1
        self.currentL = L1
        self.el = el
        self.eu = eu
        self.Ll = Ll
        self.Lu = Lu
        self.eNumber = eNumber
        self.eGrid = np.linspace(el, eu, num=eNumber)
        self.lGrid = np.array(range(Ll, Lu + 1, int(lStep)))
        self.delta = delta
        kappa = 0.2
        self.sigma = np.diag(
            [1 / ((kappa * (2))**2), 1 / ((kappa * (2))**2)])
        self.previousGamma = []

        self.allSD = []
        self.k = k
        self.K = None
        self.m = m
        self.currentData = []
        self.allData = []
        self.maxR = 1e-8
        self.a = a
        self.i = -2
        self.previous_state = None
        self.current_state = None
        np.random.seed(10)

        self.cores = cores
        self.strikes = 0
        self.maxStrikes = strikes
        self.randomSteps = randomSteps

    def calck(self, gammaI, gammaJ):
        """ Calculates the covariance k between two states

        Arguments:
            * gammaI: state 1
            * gammaJ: state 2
        Returns:
            * k: covaraiance between gammaI and gammaJ
        """
        gamma1 = (-1+2*(gammaI[0]-self.el)/(self.eu-self.el),
                  -1+2*(gammaI[1]-self.Ll)/(self.Lu-self.Ll))
        gamma2 = (-1+2*(gammaJ[0]-self.el)/(self.eu-self.el),
                  -1+2*(gammaJ[1]-self.Ll)/(self.Lu-self.Ll))

        k = np.exp(-0.5 * (np.matmul(np.transpose(gamma1),
                                     np.matmul(self.sigma, gamma2))))
        return(k)

    def calcUCB(self, testGamma):
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

        k = [None] * self.inverse.shape[0]
        for gamma, index in zip(self.previousGamma,
                                range(len(self.previousGamma))):
            k[index] = self.calck(gamma, testGamma)

        mean = np.matmul(np.transpose(k), self.inverseR) * self.s

        variance = np.matmul(self.inverse, k)
        variance = np.matmul(np.transpose(k), variance)
        variance = self.calck(testGamma, testGamma) - variance

        ucb = mean + variance * self.p * self.rootbeta

        return(ucb, mean, variance)

    def reset(self):
        """Resets the adapter"""
        self.previousGamma = []

        self.allSD = []
        self.K = None
        self.currentData = []
        self.allData = []
        self.maxR = 1e-8
        self.i = -2
        self.previous_state = None
        self.current_state = None
        self.strikes = 0

    def processChunk(self, eList, lList):
        """Processes a chunk of the e, L combinations.

        Arguments:
            * eList: list of step sizes to check
            * lList: list of leapfrog steps to check

        Returns:
            * best: a tuple of the form ((best e, best L), ucb) where the e and
            L selected are those with the highest ucb, which is also included
        """

        best = ((eList[0], lList[0]), -1000)
        for e in eList:
            for L in lList:
                ucb, mean, variance = self.calcUCB([e, L])
                if(ucb > best[1]):
                    best = ((e, L), ucb, mean, variance)
        return(best)

    def update(self, state):
        """ Steps the adapter forward by one step

        Arguments:
            * state: the newest state proposed by the HMC algorithm
        Returns:
            * currentE: the new step size
            * currentL: the new number of leapfrog steps
        """

        if(self.strikes == self.maxStrikes):
            self.el = self.el/2
            self.eu = self.eu/2
            self.eGrid = np.linspace(self.el, self.eu, num=self.eNumber)
            self.reset()
            self.strikes = 0

        self.previous_state, self.current_state = self.current_state, state

        # Calculate the square jumping distance scaled by L^(-0.5)
        if(self.previous_state is not None):
            val = 0
            for old, new in zip(self.previous_state, self.current_state):
                val += tf.math.reduce_sum(np.square(tf.reshape(new,[-1]) - tf.reshape(old,[-1]))) / (self.currentL)**(0.5)
            print("SJD:", val.numpy())
            self.currentData.append(val)
            if(val < 1e-8 and self.i // self.m > self.randomSteps):
                self.strikes += 1
            else:
                self.strikes = 0

        # Update E and L if this is not just an averaging step
        if(self.i % self.m == 0 and self.i > 0):
            u = np.random.uniform(low=0, high=1)
            self.p = max(self.i / self.m - self.k + 1, 1)**(-0.5)

            if(u < self.p):  # Over time the probability of updating will decay
                mean = np.mean(self.currentData)
                sd = np.std(self.currentData)
                self.currentData = []
                self.allData.append(mean)
                self.allSD.append(sd)
                self.maxR = max(self.allData)
                # Update the covariance matrix
                self.previousGamma.append((self.currentE, self.currentL))
                size = len(self.previousGamma)
                newK = np.ones([size, size])
                if(size > 0):
                    newK[:size - 1, :size - 1] = self.K
                for gamma, index in zip(self.previousGamma, range(
                        len(self.previousGamma))):
                    k = self.calck(gamma, self.previousGamma[-1])
                    newK[-1, index] = k
                    newK[index, -1] = k
                self.K = newK

                self.s = self.a / self.maxR  # update scalling constant

                sigmaNu = np.mean(self.allSD)  # Variance of noise

                # calculate inverse and other values only once
                try:  # In case the covaraince matrix is singular
                    self.inverse = np.linalg.inv(
                        self.K + (sigmaNu**2) * np.eye(self.K.shape[0]))
                except BaseException:
                    self.inverse = np.linalg.inv(
                        self.K + (sigmaNu**2) * np.eye(self.K.shape[0]) +
                        0.1 * np.eye(self.K.shape[0]))
                self.inverseR = np.matmul(self.inverse, self.allData)

                self.rootbeta = (self.i / self.m + 1)**(3) * math.pi**2
                self.rootbeta /= (3 * self.delta)
                self.rootbeta = np.log(self.rootbeta)*2
                self.rootbeta = self.rootbeta**(0.5)

                # Start parallel searches, take best result found
                if(self.i//self.m >= self.randomSteps):
                    # Evenly split up search space between cores
                    increment = len(self.lGrid) // self.cores
                    eList = []
                    lList = []
                    for x in range(self.cores - 1):
                        temp = self.lGrid[x * increment:(x + 1) * increment]
                        eList.append(self.eGrid)
                        lList.append(temp)
                    temp = self.lGrid[(self.cores - 1) * increment:]
                    eList.append(self.eGrid)
                    lList.append(temp)
                    best = ((self.eGrid[0], self.lGrid[0]), -1000)
                    with Pool(processes=self.cores) as pool:
                        for i in pool.starmap(
                            self.processChunk, zip(
                                eList, lList)):
                            if(i[1] > best[1]):
                                best = (i[0], i[1])

                    # Pick the state with the highest upper confidence bound
                    self.currentE = np.float32(best[0][0])
                    self.currentL = np.int64(best[0][1])
                else:
                    self.currentE = random.choice(self.eGrid)
                    self.currentL = random.choice(self.lGrid)
                    
                if(size==50):
                    self.K=self.K[1:,1:]
                    self.previousGamma=self.previousGamma[1:]
                    self.allData=self.allData[1:]
                    self.allSD=self.allSD[1:]

        self.i += 1
        return(self.currentE, self.currentL)
