import math

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

    In order to more rapidly search through the grid of possible step sizes
    and leapfrog steps this object uses parallel processing so that all
    available computing resources are used.
    """

    def __init__(self, e1, L1, el, eu, eNumber, Ll, Lu, lStep, m, k, a=4,
                 delta=0.1, cores=4):
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
        """
        self.currentE = e1
        self.currentL = L1
        self.eGrid = np.linspace(el, eu, num=500)
        self.lGrid = np.array(range(Ll, Lu + 1))
        self.delta = delta
        kappa = 0.2
        self.sigma = np.diag(
            [1 / ((kappa * (eu - el))**2), 1 / ((kappa * (Lu - Ll))**2)])
        self.previousGamma = []

        self.allSD = []
        self.k = k
        self.K = None
        self.m = m
        self.currentData = []
        self.allData = []
        self.maxR = 1
        self.a = a
        self.i = -2
        self.previous_state = None
        self.current_state = None
        np.random.seed(10)

        self.cores = cores

    def calck(self, gammaI, gammaJ):
        """ Calculates the covariance k between two states

        Arguments:
            * gammaI: state 1
            * gammaJ: state 2
        Returns:
            * k: covaraiance between gammaI and gammaJ
        """
        k = np.exp(-0.5 * (np.matmul(np.transpose(gammaI),
                                     np.matmul(self.sigma, gammaJ))))
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
        variance = self.calck(gamma, gamma) - variance

        ucb = mean + variance * self.p * self.rootbeta
        return(ucb)

    def reset(self):
        """Resets the adapter"""
        self.previousGamma = []

        self.allSD = []
        self.K = None
        self.currentData = []
        self.allData = []
        self.maxR = 1
        self.i = -2
        self.previous_state = None
        self.current_state = None

    def processChunk(self, eList, lList):
        """Processes a chunk of the e, L combinations.

        Arguments:
            * eList: list of step sizes to check
            * lList: list of leapfrog steps to check

        Returns:
            * best: a tuple of the form ((best e, best L), ucb) where the e and
            L selected are those with the highest ucb, which is also included
        """

        best = ((eList[0], lList[0]), 0)
        for e in eList:
            for L in lList:
                ucb = self.calcUCB([e, L])
                if(ucb > best[1]):
                    best = ((e, L), ucb)
        return(best)

    def update(self, state):
        """ Steps the adapter forward by one step

        Arguments:
            * state: the newest state proposed by the HMC algorithm
        Returns:
            * currentE: the new step size
            * currentL: the new number of leapfrog steps
        """
        self.previous_state, self.current_state = self.current_state, state

        # Calculate the square jumping distance scaled by L^(-1)
        if(self.previous_state is not None):
            val = 0
            for old, new in zip(self.previous_state, self.current_state):
                val += np.sum(np.square(new - old)) / (self.currentL)
            print("SJD:", str(val))
            self.currentData.append(val)

        # Update E and L if this is not just an averaging step
        if(self.i % self.m == 0 and self.i > 0):
            u = np.random.uniform(low=0, high=1)
            self.p = max(self.i / self.m - self.k + 1, 1)**(-0.5)

            if(u < self.p):  # Over time the probability of updating will decay
                newPoint = 0
                mean = np.mean(self.currentData)
                sd = np.std(self.currentData)
                self.currentData = []
                self.allData.append(mean)
                self.allSD.append(sd)
                self.maxR = max(self.maxR, newPoint)

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
                        self.K + 0.1 * np.eye(self.K.shape[0]))
                self.inverseR = np.matmul(self.inverse, self.allData)
                self.rootbeta = (self.i / self.m + 1)**(3) * math.pi**2
                self.rootbeta /= (3 * self.delta)
                self.rootbeta = np.log(self.rootbeta)*2
                self.rootbeta = self.rootbeta**(0.5)

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
                # Start parallel searches, take best result found
                best = ((self.eGrid[0], self.lGrid[0]), 0)
                with Pool(processes=self.cores) as pool:
                    for i in pool.starmap(
                        self.processChunk, zip(
                            eList, lList)):
                        if(i[1] > best[1]):
                            best = i

                # Pick the state with the highest upper confidence bound
                self.currentE = np.float32(best[0][0])
                self.currentL = np.int64(best[0][1])

                if(size == 50):
                    self.K = self.K[1:, 1:]
                    self.previousGamma = self.previousGamma[1:]
                    self.allData = self.allData[1:]
                    self.allSD = self.allSD[1:]

        self.i += 1
        return(self.currentE, self.currentL)
