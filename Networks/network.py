import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Networks.BNN_functions import multivariateLogProb
from Networks.paramAdapter import paramAdapter

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
            validateY,
            mean,
            sd):
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

        self.mean = mean
        self.sd = sd

        print(len(trainX), inputDims)
        print(trainX.shape)
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

    @tf.function
    def make_response_likelihood_regression(self, *argv, realVals=None,
                                            sd=None):
        """Make a prediction and predict its probability from a multivariate
        normal distribution

        Arguments:
            * argv: an undetermined number of tensors containg the weights
            and biases
            * realVals: the actual values for the predicted quantities
            * sd: standard deviation for output distribution, uses
            current hyper parameter value if nothing is given
        Returns:
            * result: the log probabilities of the real vals given the
            predicted values
        """

        if(sd is None):
            sd = self.hyperStates[-1]

        current = self.predict(True, argv[0])
        current = tf.transpose(current)
        sigma = tf.ones_like(current) * sd
        realVals = tf.reshape(realVals, current.shape)
        result = multivariateLogProb(sigma, current, realVals, self.dtype)

        return(result)

    @tf.function
    def make_response_likelihood_classification(self, *argv, realVals=None,
                                                sd=None):
        """Make a prediction and predict its probability from a Bernoulli
           normal distribution

        Arguments:
            * argv: an undetermined number of tensors containg the weights
                    and biases
            * realVals: the actual values for the predicted quantities
            * sd: dummy argument for compatability
        Returns:
            * result: the log probabilities of the real vals given the
                      predicted values
        """
        current = self.predict(True, argv[0])
        current = tf.cast(
            tf.clip_by_value(
                current,
                1e-8,
                1 - 1e-7),
            self.dtype)
        # Prediction distribution
        dist = tfd.Bernoulli(
            probs=current)
        result = dist.log_prob(tf.transpose(realVals))
        return(result)

    @tf.function
    def metrics(self, predictions, scaleExp, train, mean=1, sd=1):
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

        # Get the correct output values
        y = self.validateY
        if(train):
            y = self.trainY

        squaredError = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(
                predictions, tf.transpose(y)))

        scaled = tf.add(tf.multiply(tf.transpose(predictions), sd), mean)
        real = tf.add(tf.multiply(y, sd), mean)

        if(scaleExp):
            scaled = tf.exp(scaled)
            real = tf.exp(real)

        real = tf.reshape(real, scaled.shape)

        percentError = tf.reduce_mean(
            input_tensor=tf.multiply(
                tf.abs(
                    tf.divide(
                        tf.subtract(
                            scaled,
                            real),
                        real)),
                100))
        accuracy = 1 - tf.reduce_mean(tf.abs(real - tf.round(scaled)))

        return(predictions, squaredError, percentError, accuracy)

    @tf.function
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

        prob = tf.reduce_sum(
            self.make_response_likelihood(
                argv, realVals=self.trainY, sd=sd))

        # probability of the network parameters
        index = 0
        for n in range(len(self.layers)):
            numTensors = self.layers[n].numTensors
            if(numTensors > 0):
                prob += self.layers[n].calculateProbs(
                    argv[index:index + numTensors])
                index += numTensors
        return(prob)

    @tf.function
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

        if(self.regression):
            print("prob1", prob)
            prob += self.calculateProbs(self.states, sd=argv[-1])
            print("prob2", prob)

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

        @tf.function()
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
                  cores, averagingSteps=2, a=4, delta=0.1, strikes=10,
                  randomSteps=10):
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
        self.leapfrog = np.int64(leapfrog)
        self.cores = cores
        self.burnin = burnin

        self.hyper_step_size = tf.Variable(
            tf.cast(np.array(hyperStepSize), self.dtype))

        # Setup the Markov Chain for the network parameters
        self.mainKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateProbs,
            num_leapfrog_steps=self.leapfrog,
            step_size=self.step_size,
            step_size_update_fn=None,
            state_gradients_are_stopped=True)

        # Setup the Transition Kernel for the hyper parameters
        self.hyperKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateHyperProbs,
            num_leapfrog_steps=hyperLeapfrog,
            step_size=self.hyper_step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                num_adaptation_steps= int(burnin * 0.8),
                decrement_multiplier=0.01),
            state_gradients_are_stopped=True)

    @tf.function
    def updateStates(self):
        """ Updates the states of the layer object.

        Has no arguments, returns nothing.
        """
        indexh = 0
        index = 0
        for n in range(len(self.layers)):
            numHyperTensors = self.layers[n].numHyperTensors
            numTensors = self.layers[n].numTensors
            if(numHyperTensors > 0):
                self.layers[n].updateHypers(
                    self.hyperStates[indexh:indexh + numHyperTensors])
                indexh += numHyperTensors
            if(numTensors > 0):
                self.layers[n].updateParameters(
                    self.states[index:index + numHyperTensors])
                index += numTensors

    def updateKernels(self):
        """ Updates the main hamiltonian monte carlo kernel.

        Has no arguments, returns nothing.
        """
        self.step_size, self.leapfrog = self.adapt.update(self.states)
        # Setup the Markov Chain for the network parameters
        self.mainKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateProbs,
            num_leapfrog_steps=self.leapfrog,
            step_size=self.step_size,
            step_size_update_fn=None,
            state_gradients_are_stopped=True)

    def stepMCMC(self):
        """ Steps the markov chain for each of the network parameters and the
        hyper parameters forward one step

        Has no arguments, returns nothing.
        """
        num_results = 1
        # Setup the Markov Chain for the network parameters

        self.states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0,  # start collecting data on first step
            current_state=self.states,  # starting parts of chain
            parallel_iterations=self.cores,
            kernel=self.mainKernel)
        self.avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(kernel_results.log_accept_ratio,
                                0.)))
        self.loss = kernel_results.accepted_results.target_log_prob
        self.loss = -tf.reduce_mean(input_tensor=self.loss)
        for x in range(len(self.states)):
            self.states[x] = self.states[x][0]

        index = 0
        for n in range(len(self.layers)):
            numTensors = self.layers[n].numTensors
            if(numTensors > 0):
                self.layers[n].updateParameters(
                    self.states[index:index + numTensors])
                index += numTensors

        # Setup the Markov Chain for the hyper parameters
        self.hyperStates, hyper_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0,  # start collecting data on first step
            current_state=self.hyperStates,  # starting parts of chain
            parallel_iterations=self.cores,
            kernel=self.hyperKernel)
        self.hyper_avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(
                tf.minimum(hyper_kernel_results.log_accept_ratio, 0.)))
        self.hyper_loss = hyper_kernel_results.accepted_results.target_log_prob
        self.hyper_loss = -tf.reduce_mean(input_tensor=self.hyper_loss)
        for x in range(len(self.hyperStates)):
            self.hyperStates[x] = self.hyperStates[x][0]

        indexh = 0
        for n in range(len(self.layers)):
            numHyperTensors = self.layers[n].numHyperTensors
            if(numHyperTensors > 0):
                self.layers[n].updateHypers(
                    self.hyperStates[indexh:indexh + numHyperTensors])
                indexh += numHyperTensors

    def train(
            self,
            epochs,
            samplingStep,
            scaleExp=False,
            folderName=None,
            networksPerFile=1000,
            returnPredictions=False,
            regression=True):
        """Trains the network
        Arguements:
            * Epochs: Number of training cycles
            * samplingStep: Epochs between sampled networks
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

        self.regression = regression
        startSampling = self.burnin

        if self.regression:
            self.make_response_likelihood = \
                self.make_response_likelihood_regression
            self.hyperStates.append(tf.cast(0.1, self.dtype))
        else:
            self.make_response_likelihood = \
                self.make_response_likelihood_classification

        # Create the folder and files for the networks
        filePath = None
        files = []
        if(folderName is not None):
            filePath = os.path.join(os.getcwd(), folderName)
            if(not os.path.isdir(filePath)):
                os.mkdir(filePath)
            for n in range(len(self.states)):
                files.append(
                    open(
                        filePath +
                        "/" +
                        str(n) +
                        ".0" +
                        ".txt",
                        "wb"))
        with open(filePath + "/architecture.txt", "wb") as f:
            for layer in (self.layers):
                f.write((layer.name+"\n").encode("utf-8"))
        if(returnPredictions):
            self.results = []
        # get a prediction, squared error, and percent error

        iter_ = 0
        tf.random.set_seed(50)
        while(iter_ < epochs):  # Main training loop
            startTime = time.time()
            # check that the vars are not tensors
            self.stepMCMC()

            trainResult, trainSquaredError, \
                trainPercentError, trainAccuracy = self.metrics(
                    self.predict(train=True), scaleExp, True, self.mean,
                    self.sd)
            result, squaredError, percentError, accuracy = self.metrics(
                self.predict(train=False), scaleExp, False, self.mean, self.sd)

            iter_ += 1

            print()

            print(
                "iter:{:>2}  Network loss:{: 9.3f}".format(iter_, self.loss),
                "step_size:{:.7f} leapfrog_num:{:>4}".format(
                    self.step_size,
                    self.leapfrog),
                "avg_acceptance_ratio:{:.4f}".format(self.avg_acceptance_ratio)
                )
            print(
                "Hyper loss:{:9.3f}  step_size:{:.7f}".format(
                    self.hyper_loss * 1,
                    self.hyper_step_size * 1),
                "avg_acceptance_ratio:{:.4f}".format(
                    self.hyper_avg_acceptance_ratio * 1))
            if(regression):
                print(
                    "training squared error{: 9.5f}".format(trainSquaredError),
                    "training percent error{: 7.3f}".format(trainPercentError))

                print(
                    "validation squared error{: 9.5f}".format(squaredError),
                    "validation percenterror{: 7.3f}".format(percentError))
            else:
                print("training accuracy{: 9.5f}".format(trainAccuracy),
                      "validation accuracy{: 9.5f}".format(accuracy))

            self.updateKernels()

            # Create new files to record network
            indexShift = iter_ - startSampling - 1
            indexInterval = networksPerFile * samplingStep
            if(iter_ > startSampling and indexShift % indexInterval == 0):
                for file in files:
                    file.close()
                temp = []
                for n in range(len(self.states)):
                    temp.append(open(filePath +
                                     "/" +
                                     str(n) +
                                     "." +
                                     str(int(iter_ //
                                             (networksPerFile *
                                              samplingStep))) +
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
                            " " + str(len(self.states))).encode("utf-8"))
                file.close()

            # Record prediction
            if(iter_ > startSampling and (iter_) % samplingStep == 0):
                if(returnPredictions):
                    self.results.append(result_)
                if(filePath is not None):
                    for n in range(len(files)):
                        np.savetxt(files[n], self.states[n])
            if(self.regression):
                print("Loss Standard Deviation",
                      self.hyperStates[-1].numpy()*1)
            print("Time elapsed:", time.time() - startTime)

        for file in files:
            file.close()
        if(returnPredictions):
            return(self.results)
