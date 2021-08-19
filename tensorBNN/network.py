import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorBNN.paramAdapter import paramAdapter

tfd = tfp.distributions


class network(object):
    """An object used for storing all of the variables required to create
    a Bayesian Neural Network using Hamiltonian Monte Carlo and then training
    the network.
    """

    def __init__(
            self,
            dtype,
            inputDims,
            trainX,
            trainY,
            validateX,
            validateY):
        """
        Arguments:
            * dtype: data type for Tensors
            * inputDims: dimension of input vector
            * trainX: the training data input, shape is n by inputDims
            * trainY: the training data output
            * validateX: the validation data input, shape is n by inputDims
            * validateY: the validation data output
            * mean: the mean used to scale trainY and validateY
            * sd: standard deviation used to scale trainY and validateY
        """
        self.dtype = dtype
        self.iteration = None

        self.trainX = tf.reshape(
            tf.constant(
                trainX, dtype=self.dtype), [
                len(trainX), inputDims])
        self.trainY = tf.constant(trainY, dtype=self.dtype)

        self.validateX = tf.reshape(
            tf.constant(
                validateX, dtype=self.dtype), [
                len(validateX), inputDims])
        self.validateY = tf.constant(validateY, dtype=self.dtype)

        self.states = []  # List with the weight and bias state placeholders
        self.hyperStates = []  # List with hyper parameter state placeholders

        self.layers = []  # List of all the layers

        self.currentInnerStep = None

    def metrics(self, trainPredict, trainReal, validatePredict, validateReal):
        """Calculates the average squared error and percent difference of the
        current network
        Arguments:
            * predictions: output from the network
            * scaleExp: boolean value to determine whether to take the
                        exponential of the data and scale it
            * train: boolean value to determine whether to use the training
                     data
            * mean: mean value used for unshifiting a distribution
            * sd: sd value used for unscalling a distribution
        Returns:
            * logits: output from the network
            * squaredError: the mean squared error of predictions from the
                            network
            * percentError: the percent error of the predictions from the
                            network
        """

        for metric in self.metricList:
            metric.calculate(trainPredict, validatePredict, trainReal,
                             validateReal)
            metric.display()

    def calculateProbs(self, *argv, sd=None):
        """Calculates the log probability of the current network values
        as well as the log probability of their prediction.
        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases.
            * sd: standard deviation for output distribution, uses current
            hyper value if none is given
        Returns:
            * prob: log probability of network values and network prediction
        """
        if(len(argv) != len(self.states)):
            argv = argv[0]

        temp = self.makeResponseLikelihood(argv, predict=self.predict,
                                           dtype=self.dtype,
                                           hyperStates=self.hyperStates,
                                           realVals=self.trainY, sd=sd)
        prob = tf.reduce_sum(temp)

        # probability of the network parameters
        index = 0
        for n in range(len(self.layers)):
            numTensors = self.layers[n].numTensors
            if(numTensors > 0):
                prob += self.layers[n].calculateProbs(
                    argv[index:index + numTensors])
                index += numTensors
        return(prob)

    def calculateHyperProbs(self, *argv):
        """Calculates the log probability of the current hyper parameters
        Arguments:
            * argv: an undetermined number of tensors containg the hyper
                    parameters
        Returns:
            * prob: log probability of hyper parameters given their priors
        """
        prob = 0
        indexh = 0
        index = 0
        for n in range(len(self.layers)):
            numHyperTensors = self.layers[n].numHyperTensors
            numTensors = self.layers[n].numTensors
            if(numHyperTensors > 0):

                prob += self.layers[n].calculateHyperProbs(
                    argv[indexh:indexh + numHyperTensors],
                    self.states[index:index + numTensors])
                indexh += numHyperTensors
                index += numTensors

        if(self.likelihood.mainProbsInHypers):
            prob += self.calculateProbs(self.states, sd=argv[-1])

        return(prob)

    def predict(self, train, *argv):
        """Makes a prediction
        Arguments:
            * train: a boolean value which determines whether to use training
                     data
            * argv: an undetermined number of tensors containg the weights
            and biases.
        Returns:
            * prediction: a prediction from the network
        """
        tensors = argv
        if(len(tensors) == 0):
            tensors = self.states
        else:
            tensors = tensors[0]
        x = self.trainX
        if(not train):
            x = self.validateX

        def innerPrediction(x, layers):
            prediction = tf.transpose(a=x)
            index = 0
            for n in range(len(self.layers)):
                numTensors = layers[n].numTensors
                prediction = layers[n].predict(
                    prediction, tensors[index:index + numTensors])
                index += numTensors
            return(prediction)
        prediction = innerPrediction(x, self.layers)

        return(prediction)

    def add(self, layer, parameters=None):
        """Adds a new layer to the network
        Arguments:
            * layer: the layer to be added
            * parameters: list containing weight, bias, and acitvation
            matrices
        """
        self.layers.append(layer)
        if(layer.numTensors > 0):
            if parameters is None:
                for states in layer.parameters:
                    self.states.append(states)
            else:
                for states in parameters:
                    self.states.append(states)

        if(layer.numHyperTensors > 0):
            for states in layer.hypers:
                self.hyperStates.append(states)

    def setupMCMC(self, stepSize, stepMin, stepMax, stepNum, leapfrog, leapMin,
                  leapMax, leapStep, hyperStepSize, hyperLeapfrog, burnin,
                  cores, averagingSteps=10, a=4, delta=0.1, strikes=5,
                  randomSteps=10, dualAveraging=False):
        """Sets up the MCMC algorithms
        Arguments:
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
            * strikes: iterations with no movement before reseting adapter
            * randomSteps: averaging cycles at beginning with random values
        Returns nothing
        """

        # Adapt the step size and number of leapfrog steps
        self.adapt = paramAdapter(
            stepSize,
            leapfrog,
            stepMin,
            stepMax,
            stepNum,
            leapMin,
            leapMax,
            leapStep,
            averagingSteps,
            burnin / averagingSteps,
            a=a,
            delta=delta,
            cores=cores,
            strikes=strikes,
            randomSteps=randomSteps)

        self.step_size = tf.cast(stepSize, self.dtype)
        self.leapfrog = tf.cast(leapfrog, tf.int32)
        self.cores = cores
        self.burnin = burnin
        self.target = 0.95

        self.gamma = tf.cast(0.4, self.dtype)
        self.t0 = tf.cast(10, self.dtype)
        self.kappa = tf.cast(0.75, self.dtype)
        self.h = tf.cast([0], self.dtype)
        self.logEpsilonBar = tf.cast([0], self.dtype)
        self.mu = tf.cast(tf.math.log(100*hyperStepSize), self.dtype)

        self.dualAveraging = dualAveraging
        self.gamma2 = tf.cast(0.4, self.dtype)
        self.t02 = tf.cast(10, self.dtype)
        self.kappa2 = tf.cast(0.75, self.dtype)
        self.h2 = tf.cast([0], self.dtype)
        self.logEpsilonBar2 = tf.cast([0], self.dtype)
        self.mu2 = tf.cast(tf.math.log(100*stepSize), self.dtype)

        self.hyper_step_size = tf.Variable(tf.cast(np.array(hyperStepSize),
                                                   self.dtype))

        # Setup the Markov Chain for the network parameters
        self.mainKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateProbs,
            num_leapfrog_steps=self.leapfrog,
            step_size=[self.step_size],
            state_gradients_are_stopped=True,
            name="main")

        self.hyperLeapfrog = hyperLeapfrog
        # Setup the Transition Kernel for the hyper parameters
        hyperKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateHyperProbs,
            num_leapfrog_steps=hyperLeapfrog,
            step_size=[self.hyper_step_size],
            state_gradients_are_stopped=True)

        self.hyperKernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hyperKernel, num_adaptation_steps=int(burnin * 0.8))

    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def stepMCMCNoHypers(self, states, hyperStates, mainStep, leapfrogVal,
                         mainAccept=tf.cast([1], tf.float32),  sampleNumber=1):
        """ Steps the markov chain for each of the network parameters and the
        hyper parameters forward one step
        Has no arguments, returns nothing.
        """

        def InnerStepMain(i, states, hyperStates, leapfrog, step_size):

            def calculateProbs(*argv):
                if(len(argv) != len(self.states)):
                    argv = argv[0]

                prob = 0
                indexh = 0
                index = 0
                for n in range(len(self.layers)):
                    numHyperTensors = self.layers[n].numHyperTensors
                    numTensors = self.layers[n].numTensors
                    if(numHyperTensors > 0):

                        prob += self.layers[n].calculateProbs(
                            hyperStates[indexh:indexh + numHyperTensors],
                            argv[index:index + numTensors])
                        indexh += numHyperTensors
                        index += numTensors

                temp = self.makeResponseLikelihood(argv, predict=self.predict,
                                                   dtype=self.dtype,
                                                   hyperStates=hyperStates,
                                                   realVals=self.trainY,
                                                   sd=hyperStates[-1])
                prob += tf.reduce_sum(temp)
                return(prob)
            hmc = tfp.mcmc.HamiltonianMonteCarlo
            kernel = hmc(target_log_prob_fn=calculateProbs,
                         num_leapfrog_steps=leapfrog,
                         step_size=step_size,
                         state_gradients_are_stopped=True)

            states, kernel_results = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=0,  # start collecting data on first step
                current_state=states,  # starting parts of chain
                parallel_iterations=8,
                kernel=kernel,
                trace_fn=lambda _, pkr: [pkr.accepted_results.step_size,
                                         pkr.log_accept_ratio,
                                         pkr.accepted_results.target_log_prob])

            acceptRate = tf.where(kernel_results[1] < 0,
                                  tf.exp(kernel_results[1]), 1)

            return(states, [step_size], acceptRate)

        def oneStep(i, params, hyperParams, mainStep, mainAccept, leap):

            params, mainStep, mainAccept = InnerStepMain(i, params,
                                                         hyperParams, leap,
                                                         mainStep)
            for x in range(len(params)):
                params[x] = params[x][0]

            return(tf.add(i, 1), params, hyperParams, mainStep[0], mainAccept,
                   leap)

        def condition(i, states, hyperStates, mainStep, mainAccept,
                      leapfrogVal):
            return(tf.less(i, sampleNumber))

        i = tf.constant(0)
        i, states, hyperStates, mainStep, mainAccept, \
            leapfrogVal = tf.while_loop(condition, oneStep,
                                        [i, states, hyperStates, mainStep,
                                         mainAccept, leapfrogVal])

        return(states, mainStep, mainAccept)

    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def stepMCMC(self, states, hyperStates, mainStep, hyperStep, logEpsilonBar,
                 h, iter_, leapfrogVal, mainAccept=tf.cast([1], tf.float32),
                 hyperAccept=tf.cast([1], tf.float32),  sampleNumber=1):
        """ Steps the markov chain for each of the network parameters and the
        hyper parameters forward one step
        Has no arguments, returns nothing.
        """

        def InnerStepMain(i, states, hyperStates, leapfrog, step_size, epoch):

            def calculateProbs(*argv):
                if(len(argv) != len(self.states)):
                    argv = argv[0]

                prob = 0
                indexh = 0
                index = 0
                for n in range(len(self.layers)):
                    numHyperTensors = self.layers[n].numHyperTensors
                    numTensors = self.layers[n].numTensors
                    if(numHyperTensors > 0):

                        prob += self.layers[n].calculateProbs(
                            hyperStates[indexh:indexh + numHyperTensors],
                            argv[index:index + numTensors])
                        indexh += numHyperTensors
                        index += numTensors

                prob += tf.reduce_sum(self.makeResponseLikelihood(
                    argv, predict=self.predict, dtype=self.dtype,
                    hyperStates=hyperStates, realVals=self.trainY,
                    sd=hyperStates[-1]))
                return(prob)

            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=calculateProbs,
                num_leapfrog_steps=leapfrog,
                step_size=step_size,
                state_gradients_are_stopped=True)

            states, kernel_results = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=0,  # start collecting data on first step
                current_state=states,  # starting parts of chain
                parallel_iterations=8,
                kernel=kernel,
                trace_fn=lambda _, pkr: [pkr.accepted_results.step_size,
                                         pkr.log_accept_ratio,
                                         pkr.accepted_results.target_log_prob])

            acceptRate = tf.where(kernel_results[1] < 0,
                                  tf.exp(kernel_results[1]), 1)
            return(states, [step_size], acceptRate)

        def InnerStepHyper(i, states, hyperStates, leapfrog, step_size,
                           logEpsilonBar, h, epoch):

            def calculateProbs(*argv):
                if(len(argv) != len(self.hyperStates)):
                    argv = argv[0]

                prob = 0
                indexh = 0
                index = 0
                for n in range(len(self.layers)):
                    numHyperTensors = self.layers[n].numHyperTensors
                    numTensors = self.layers[n].numTensors
                    if(numHyperTensors > 0):

                        prob += self.layers[n].calculateHyperProbs(
                            argv[indexh:indexh + numHyperTensors],
                            states[index:index + numTensors])
                        indexh += numHyperTensors
                        index += numTensors

                if(self.likelihood.mainProbsInHypers):
                    prob += tf.reduce_sum(self.makeResponseLikelihood(
                        states, predict=self.predict, dtype=self.dtype,
                        hyperStates=argv, realVals=self.trainY, sd=argv[-1]))

                return(prob)

            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=calculateProbs,
                num_leapfrog_steps=leapfrog,
                step_size=step_size,
                state_gradients_are_stopped=True)

            hyperStates, kernel_results = tfp.mcmc.sample_chain(
                num_results=1,
                num_burnin_steps=0,  # start collecting data on first step
                current_state=hyperStates,  # starting parts of chain
                parallel_iterations=8,
                kernel=kernel,
                trace_fn=lambda _, pkr: [pkr.accepted_results.step_size,
                                         pkr.log_accept_ratio,
                                         pkr.accepted_results.target_log_prob])
            m = epoch + 1

            accept = tf.where(kernel_results[1] < 0,
                              tf.exp(kernel_results[1]), 1)
            h = (1-1/(m+self.t0))*h+(1/(m+self.t0))*(self.target-accept)

            logEpsilon = self.mu-h*(m**0.5)/self.gamma

            logEpsilonBar = (1-m**(-self.kappa))*logEpsilonBar
            logEpsilonBar += m**(-self.kappa)*logEpsilon

            step_size = tf.where(m < self.burnin * 0.8,
                                 tf.math.exp(logEpsilonBar), step_size)

            return(hyperStates, step_size, logEpsilonBar, h, accept)

        def oneStep(i, params, hyperParams, mainStep, hyperStep, logEpsilonBar,
                    h, epoch, mainAccept, hyperAccept, leap):

            params, mainStep, mainAccept = InnerStepMain(i, params,
                                                         hyperParams, leap,
                                                         mainStep, epoch)
            for x in range(len(params)):
                params[x] = params[x][0]

            hyperParams, hyperStep, logEpsilonBar, h, hyperAccept = \
                InnerStepHyper(i, params, hyperParams, self.hyperLeapfrog,
                               hyperStep, logEpsilonBar, h, epoch)
            for x in range(len(hyperParams)):
                hyperParams[x] = hyperParams[x][0]
            hyperStep = hyperStep[0]

            return(tf.add(i, 1), params, hyperParams, mainStep[0], hyperStep,
                   logEpsilonBar, h, epoch, mainAccept, hyperAccept, leap)

        def condition(i, states, hyperStates, mainStep, hyperStep,
                      logEpsilonBar, h, epoch, mainAccept, hyperAccept, leap):
                        return(tf.less(i, sampleNumber))

        i = tf.constant(0)
        epoch = tf.cast(iter_, self.dtype)
        i, states, hyperStates, mainStep, hyperStep, logEpsilonBar, h, epoch, \
            mainAccept, hyperAccept, leapfrogVal = \
            tf.while_loop(condition, oneStep, [i, states, hyperStates,
                                               mainStep, hyperStep,
                                               logEpsilonBar, h, epoch,
                                               mainAccept, hyperAccept,
                                               leapfrogVal])

        return(states, hyperStates, mainStep, hyperStep, logEpsilonBar, h,
               mainAccept, hyperAccept)

    def train(
            self,
            epochs,
            samplingStep,
            likelihood,
            metricList=[],
            adjustHypers=True,
            scaleExp=False,
            folderName=None,
            networksPerFile=1000,
            displaySkip=1):
        """Trains the network
        Arguements:
            * Epochs: Number of training cycles
            * samplingStep: Epochs between sampled networks
            * likelihood: Object containing the output likelihood for the BNN
            * scaleExp: whether the metrics should be scaled via exp
            * folderName: name of folder for saved networks
            * networksPerFile: number of networks saved in a given file
            * returnPredictions: whether to return the prediction from the
                                 network
        Returns:
            * results: the output of the network when sampled
                       (if returnPrediction=True)
        """
        # Create response likelihood
        startSampling = self.burnin

        self.likelihood = likelihood
        self.makeResponseLikelihood = self.likelihood.makeResponseLikelihood
        self.metricList = metricList
        self.adjustHypers = adjustHypers

        for val in self.likelihood.hypers:
            self.hyperStates.append(tf.cast(val, self.dtype))

        # Create the folder and files for the networks
        filePath = None
        files = []
        if(folderName is not None):
            filePath = os.path.join(os.getcwd(), folderName)
            if(not os.path.isdir(filePath)):
                os.mkdir(filePath)
            for n in range(len(self.states)):
                files.append(
                    open(filePath + "/" + str(n) + ".0" + ".txt", "wb"))
            files.append(open(filePath + "/hypers" + "0" + ".txt", "wb"))
        previousLeap = self.leapfrog
        with open(filePath + "/architecture.txt", "wb") as f:
            for layer in (self.layers):
                f.write((layer.name+"\n").encode("utf-8"))

        iter_ = 0
        tf.random.set_seed(50)

        self.mainAccept = tf.cast([0], tf.float32)
        self.hyperAccept = tf.cast([0], tf.float32)
        startTime = time.time()
        while(iter_ < epochs):  # Main training loop
            #
            if(self.adjustHypers):
                returnVals = self.stepMCMC(self.states, self.hyperStates,
                                           self.step_size,
                                           self.hyper_step_size,
                                           self.logEpsilonBar, self.h,
                                           tf.cast(iter_, self.dtype),
                                           self.leapfrog,
                                           tf.cast(self.mainAccept,
                                                   self.dtype),
                                           tf.cast(self.hyperAccept,
                                                   self.dtype))
                self.states, self.hyperStates, self.step_size, \
                    self.hyper_step_size, self.logEpsilonBar, self.h, \
                    self.mainAccept, self.hyperAccept = returnVals

            else:
                self.states, self.step_size, self.mainAccept = \
                    self.stepMCMCNoHypers(self.states, self.hyperStates,
                                          self.step_size, self.leapfrog,
                                          tf.cast(self.mainAccept, self.dtype))

            previousLeap = self.leapfrog
            iter_ += 1

            if(iter_ % displaySkip == 0):
                print()
                print("iter:{:>2}".format(iter_))
                print("step size", self.step_size.numpy())
                print("hyper step size", self.hyper_step_size.numpy())
                print("leapfrog", self.leapfrog.numpy())
                print("Main acceptance", self.mainAccept.numpy()[0])
                print("Hyper acceptance", self.hyperAccept.numpy()[0])
                self.metrics(self.predict(train=True), self.trainY,
                             self.predict(train=False), self.validateY)
            step, leap = self.adapt.update(self.states)
            self.step_size = step + self.step_size * 0
            self.leapfrog = leap + self.leapfrog * 0

            self.step_size = tf.cast(self.step_size, self.dtype)

            # Create new files to record network
            indexShift = iter_ - startSampling - 1
            indexInterval = networksPerFile * samplingStep
            if(iter_ > startSampling and indexShift % indexInterval == 0):
                for file in files:
                    file.close()
                temp = []
                for n in range(len(self.states)):
                    temp.append(open(filePath + "/" + str(n) + "." +
                                     str(int((iter_-startSampling) //
                                             (networksPerFile *
                                              samplingStep))) +
                                     ".txt", "wb"))
                temp.append(open(filePath + "/hypers" +
                                 str(int((iter_-startSampling) //
                                     (networksPerFile * samplingStep))) +
                                 ".txt", "wb"))
                files = temp

                # Update the summary file
                file = open(filePath + "/summary.txt", "wb")
                for n in range(len(self.states)):
                    val = ""
                    for sizes in self.states[n].shape:
                        val += str(sizes) + " "
                    val = val.strip() + "\n"
                    file.write(val.encode("utf-8"))
                numNetworks = (indexShift) // samplingStep
                numFiles = numNetworks // networksPerFile
                if(numNetworks % networksPerFile != 0):
                    numFiles += 1
                file.write((str(numNetworks) + " " + str(numFiles) +
                            " " + str(len(self.states))+"\n").encode("utf-8"))
                hyperStateCount = 0
                for state in self.hyperStates:
                    hyperStateCount += tf.size(state)
                file.write(str(hyperStateCount.numpy()).encode("utf-8"))
                file.close()
            # Record prediction
            if(iter_ > startSampling and (iter_) % samplingStep == 0):
                if(filePath is not None):
                    for n in range(len(files)-1):
                        np.savetxt(files[n], self.states[n])
                    tempStates = []
                    for state in self.hyperStates:
                        length = 1
                        for x in state.shape:
                            length = length*x
                        if(length > 1):
                            splitStates = tf.split(state, length)
                            for splitState in splitStates:
                                tempStates.append(splitState)
                        else:
                            tempStates.append(state)
                    np.savetxt(files[-1], tempStates)
            if(iter_ % displaySkip == 0):
                likelihood.display(self.hyperStates)
                print("Time elapsed:", time.time() - startTime)
                startTime = time.time()

        for file in files:
            file.close()
