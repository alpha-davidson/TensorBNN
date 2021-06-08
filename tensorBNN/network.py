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

        prob = tf.reduce_sum(
            self.makeResponseLikelihood(
                argv, predict=self.predict, dtype=self.dtype, 
                hyperStates=self.hyperStates, realVals=self.trainY, sd=sd))

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
            step_size=[self.step_size],
            state_gradients_are_stopped=True,
            name="main")
        """
        self.mainKernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=mainKernel, num_adaptation_steps=int(burnin * 0.8))
        """
        
        # Setup the Transition Kernel for the hyper parameters
        hyperKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateHyperProbs,
            num_leapfrog_steps=hyperLeapfrog,
            step_size=[self.hyper_step_size],
            state_gradients_are_stopped=True)
        
        self.hyperKernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hyperKernel, num_adaptation_steps=int(burnin * 0.8))

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

        
    @tf.function(jit_compile=True)
    def innerStepMain(self, leapfrogSteps, stepSize, states, kernel_results):
        
        """
        mainKernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.calculateProbs,
            num_leapfrog_steps=leapfrogSteps,
            step_size=stepSize,
            state_gradients_are_stopped=True)
        """
        """
        
        num_results = 1
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0,  # start collecting data on first step
            current_state=states,  # starting parts of chain
            parallel_iterations=1,
            kernel=mainKernel,
            trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio,
                             pkr.inner_results.accepted_results.target_log_prob])
        """
        """
        kernelInnerAccepted = kernel_results.inner_results.accepted_results._replace(
                    num_leapfrog_steps = kernel_results.inner_results.accepted_results.num_leapfrog_steps+1)
        kernelInner =  kernel_results.inner_results._replace(accepted_results=kernelInnerAccepted)
        kernel_results = kernel_results._replace(inner_results=kernelInner)
        """
        states, kernel_results = self.mainKernel.one_step(states, kernel_results)
        
        return([kernel_results.accepted_results.step_size, kernel_results.log_accept_ratio,
                 kernel_results.accepted_results.target_log_prob], states,kernel_results)
    
    @tf.function(jit_compile=True)
    def innerStepHyper(self, hyperStates,kernel_results):
        """
        num_results=1
        hyperStates, hyper_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=0,  # start collecting data on first step
            current_state=hyperStates,  # starting parts of chain
            parallel_iterations=1,#self.cores,
            kernel=self.hyperKernel,
            trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio,
                             pkr.inner_results.accepted_results.target_log_prob])
        """
        
        
        hyperStates, kernel_results = self.hyperKernel.one_step(hyperStates, kernel_results)
        
        return([kernel_results.new_step_size, kernel_results.inner_results.log_accept_ratio,
                 kernel_results.inner_results.accepted_results.target_log_prob], hyperStates,kernel_results)
    
    
    def stepMCMC(self):
        """ Steps the markov chain for each of the network parameters and the
        hyper parameters forward one step
        Has no arguments, returns nothing.
        """

        # Setup the Markov Chain for the network parameters
        [step_size, log_accept_ratio, target_log_prob], self.states,\
            self.mainKernelStatus = self.innerStepMain(self.leapfrog, self.step_size, self.states,self.mainKernelStatus)
        func = self.innerStepMain._stateful_fn._get_concrete_function_internal_garbage_collected(self.leapfrog, self.step_size, self.states,self.mainKernelStatus)
        self.step_size=step_size
        self.leapfrog = self.mainKernelStatus.accepted_results.num_leapfrog_steps
        if(not(self.currentInnerStep is func)):
            del self.currentInnerStep
            self.currentInnerStep = func
        self.avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(tf.minimum(log_accept_ratio,
                                0.)))
        self.loss = target_log_prob
        
        self.loss = -tf.reduce_sum(input_tensor=self.loss)
        
        index = 0
        for n in range(len(self.layers)):
            numTensors = self.layers[n].numTensors
            if(numTensors > 0):
                self.layers[n].updateParameters(
                    self.states[index:index + numTensors])
                index += numTensors
        
        # Setup the Markov Chain for the hyper parameters
        [step_size, log_accept_ratio, target_log_prob], self.hyperStates,\
            self.hyperKernelStatus= self.innerStepHyper(self.hyperStates,self.hyperKernelStatus)
        
        self.hyper_avg_acceptance_ratio = tf.reduce_mean(
            input_tensor=tf.exp(
                tf.minimum(log_accept_ratio, 0.)))
        
        self.hyper_loss = target_log_prob
        self.hyper_step_size = step_size[0]
        self.hyper_loss = -tf.reduce_mean(input_tensor=self.hyper_loss)
        
        """
        for x in range(len(self.hyperStates)):
            self.hyperStates[x] = self.hyperStates[x][0]
        
        self.hyper_loss = 1
        self.hyper_step_size = 1
        self.hyper_loss = 1
        self.hyper_avg_acceptance_ratio=1
        """
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
            likelihood,
            metricList=[],
            scaleExp=False,
            folderName=None,
            networksPerFile=1000):
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
        self.metricList=metricList

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
        with open(filePath + "/architecture.txt", "wb") as f:
            for layer in (self.layers):
                f.write((layer.name+"\n").encode("utf-8"))

        iter_ = 0
        tf.random.set_seed(50)
        self.mainKernelStatus = self.mainKernel.bootstrap_results(self.states)
        self.hyperKernelStatus = self.hyperKernel.bootstrap_results(self.hyperStates)
        kernelAccepted = self.mainKernelStatus.accepted_results._replace(
                    num_leapfrog_steps = self.leapfrog, step_size = self.step_size)
        self.mainKernelStatus = self.mainKernelStatus._replace(accepted_results=kernelAccepted)
        startTime = time.time()
        while(iter_ < epochs):  # Main training loop
            iter_ += 1
            self.stepMCMC()
            
            if(iter_%1==0):
                
                print()
                print(self.step_size,
                        self.leapfrog)
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
                self.metrics(self.predict(train=True), self.trainY,
                             self.predict(train=False), self.validateY )
            step, leap = self.adapt.update(self.states)
            self.step_size=step+self.step_size*0
            self.leapfrog=leap
            kernelAccepted = self.mainKernelStatus.accepted_results._replace(
                    num_leapfrog_steps = self.leapfrog, step_size = self.step_size)
            self.mainKernelStatus = self.mainKernelStatus._replace(accepted_results=kernelAccepted)
            
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
                temp.append(open(filePath + "/hypers" + str(int((iter_-startSampling) //
                                             (networksPerFile *
                                              samplingStep))) + ".txt", "wb"))
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
                  hyperStateCount+=tf.size(state)
                file.write(str(hyperStateCount.numpy()).encode("utf-8"))
                file.close()
            # Record prediction
            if(iter_ > startSampling and (iter_) % samplingStep == 0):
                if(filePath is not None):
                    for n in range(len(files)-1):
                        np.savetxt(files[n], self.states[n])
                    tempStates=[]
                    for state in self.hyperStates:
                     length=1
                     for x in state.shape:
                        length=length*x
                     if(length>1):
                       splitStates = tf.split(state, length)
                       for splitState in splitStates:
                           tempStates.append(splitState)
                     else:
                       tempStates.append(state)
                    np.savetxt(files[-1], tempStates)
            if(iter_%1==0):
                likelihood.display(self.hyperStates)           
                print("Time elapsed:", time.time() - startTime)
                startTime = time.time()

        for file in files:
            file.close()
