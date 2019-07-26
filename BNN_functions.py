import math

import numpy as np
import tensorflow as tf

from scipy import stats


def normalizeData(trainIn, trainOut, valIn, valOut, log=True):
    """Normalizes the training and validation data to improve network training.

    The output data is normalized by taking its log and then scaling according
    to its normal distribution fit. The input data is normalized by scaling
    it down to [-1,1] using its min and max.

    Inputs:
        * trainIn: Numpy array containing the training input data
        * trainOut: Numpy array containing the training output data
        * valIn: Numpy array containing the validation input data
        * valOut: Numpy array containing the validation output data
        * log: Whether to log normalize data

    Returns:
        * data: List containing the normalized input data in the same order
        * normInfo: List containing values required to un-normalize the data
                    Of the form: [(output_mean, output_sd),
                                  (input1_min, input1_max),
                                  (input2_min, input2_max), ...]
    """

    # Copy data
    trainIn = np.copy(trainIn)
    trainOut = np.copy(trainOut)
    valIn = np.copy(valIn)
    valOut = np.copy(valOut)

    normInfo = []  # stores the data required to un-normalize the data

    # Take the log of the output distributions
    if(log):
        trainOutput = np.log(trainOut[:, 0])
        valOutput = np.log(valOut[:, 0])
    else:
        trainOutput = trainOut[:, 0]
        valOutput = valOut[:, 0]
    # Combine the output from the train and validation
    fullOutput = trainOutput.tolist() + valOutput.tolist()

    # Calculate the mean and standard deviation for the output
    mean, sd = stats.norm.fit(fullOutput)

    # Scale the output

    trainOutput -= mean
    trainOutput /= sd
    valOutput -= mean
    valOutput /= sd

    # Save the mean and standard deviation
    normInfo.append((mean, sd))

    # Scale all the input data from -1 to 1
    for x in range(len(trainIn[1, :])):
        minVal = min(np.amin(trainIn[:, x]), np.amin(valIn[:, x]))
        maxVal = max(np.amax(trainIn[:, x]), np.amax(valIn[:, x]))
        trainIn[:, x] = (trainIn[:, x] - minVal) * 2 / (maxVal - minVal) - 1
        valIn[:, x] = (valIn[:, x] - minVal) * 2 / (maxVal - minVal) - 1

        # Save the min and max
        normInfo.append((minVal, maxVal))

    # Combine the data into a single list
    data = [trainIn, trainOutput, valIn, valOutput]

    return(normInfo, data)


@tf.function
def multivariateLogProb(sigmaIn, mu, x, dtype=tf.float32):
    """ Calculates the log probability of x given mu and sigma defining
    a multivariate normal distribution.

    Arguments:
        * sigmaIn: an n-dimensional vector with the standard deviations of
        * the distribution
        * mu: an n-dimensional vector with the means of the distribution
        * x: m n-dimensional vectors to have their probabilities calculated
        * dtype: data type of calculation
    Returns:
        * prob: an m-dimensional vector with the log-probabilities of x
    """
    sigma = sigmaIn

    sigma = tf.maximum(sigma, tf.cast(10**(-8), dtype))
    sigma = tf.minimum(sigma, tf.cast(10**(8), dtype))
    logDet = 2 * tf.reduce_sum(input_tensor=tf.math.log(sigma))
    k = tf.size(input=sigma, out_type=dtype)
    inv = tf.divide(1, sigma)
    difSigma = tf.math.multiply(inv, tf.subtract(x, mu))
    difSigmaSquared = tf.reduce_sum(tf.math.multiply(difSigma, difSigma))
    twoPi = tf.cast(2 * math.pi, dtype)

    logLikelihood = -0.5 * (logDet + difSigmaSquared + k * tf.math.log(twoPi))

    return(logLikelihood)


@tf.function
def cauchyLogProb(gamma, x0, x, dtype=tf.float32):
    """ Calculates the log probability of x given mu and sigma defining
    a multivariate normal distribution.

    Arguments:
        * sigma: an n-dimensional vector with the standard deviations of
        * the distribution
        * mu: an n-dimensional vector with the means of the distribution
        * x: m n-dimensional vectors to have their probabilities calculated
        * dtype: data type of calculation
    Returns:
        * prob: an m-dimensional vector with the log-probabilities of x
    """

    a = tf.math.log(1 + ((x - x0) / gamma)**2)
    b = tf.math.log(tf.cast(math.pi * gamma, dtype))
    c = tf.ones_like(x)
    d = -tf.math.scalar_mul(b, c)
    prob = a + d
    prob = tf.cast(prob, dtype)
    return(prob)


def loadNetworks(directoryPath):
    """Loads saved networks.

    Arguments:
        * directoryPath: the path to the directory where the networks are saved
    Returns:
        * numNetworks: Total number of networks
        * numMatrices: Number of matrices in the network
        * matrices: A list containing all the extracted matrices
    """

    summary = []
    with open(directoryPath + "summary.txt", "r") as file:
        for line in iter(file):
            summary.append(line.split())

    numNetworks = int(summary[-1][0])
    numMatrices = int(summary[-1][2])
    numFiles = int(summary[-1][1])

    numNetworks //= numFiles

    matrices = []
    for n in range(numMatrices):
        weightsSplitDims = (numNetworks *
                            numFiles, int(summary[n][0]), int(summary[n][1]))
        weights0 = np.zeros(weightsSplitDims)
        for m in range(numFiles):
            weights = np.loadtxt(
                directoryPath +
                str(n) +
                "." +
                str(m) +
                ".txt",
                dtype=np.float32,
                ndmin=2)
            for k in range(numNetworks):
                weights0[m *
                         numNetworks +
                         k, :, :] = weights[weightsSplitDims[1] *
                                            k:weightsSplitDims[1] *
                                            (k +
                                             1), :weightsSplitDims[2]]
        matrices.append(weights0)
    numNetworks *= numFiles
    return(numNetworks, numMatrices, matrices)


def predict(inputMatrix, numNetworks, numMatrices, matrices):
    """Make predictions from an ensemble of neural networks.

    Arguments:
        * inputMatrix: The input data
    Returns:
        * numNetworks: Number of networks used
        * numMatrices: Number of matrices in the network
        * matrices: List with all networks used
    """

    inputVal = np.transpose(inputMatrix)
    initialResults = [None] * (numNetworks)
    for m in range(numNetworks):
        current = inputVal
        for n in range(0, numMatrices, 2):
            current = np.matmul(matrices[n][m, :, :], current)
            current += matrices[n + 1][m, :, :]
            if(n + 2 < numMatrices):
                current = np.maximum(current, 0)
        if(m % 100 == 0):
            print(m / numNetworks / numFiles)
        initialResults[m] = current

    return(initialResults)


def trainBasicRegression(
        hidden,
        inputDims,
        outputDims,
        width,
        cycles,
        epochs,
        alpha,
        patience,
        trainIn,
        trainOut,
        valIn,
        valOut,
        name):
    """Trains a basic regression neural network and returns its weights. Uses
    the amsgrad optimizer and a learning rate of 0.01 which decays by a factor
    of 10 each cycle. The activation function is PReLU. Saves the network as
    name in case something goes wrong with the BNN code so the network does
    not need to be retrained.

    Arguments:
        * hidden: number of hidden layers
        * inputDims: input dimension
        * outputDims: output dimension
        * width: width of hidden layers
        * cycles: number of training cycles with decaying learning rates
        * epochs: number of epochs per cycle
        * alpha: slope value for leaky ReLU
        * patience: early stopping patience
        * trainIn: training input data
        * trainOut: training output data
        * valIn: validation input data
        * valOut: validation output data
        * name: name of network
    Returns:
        * weights: list containing all weight matrices
        * biases: list containing all bias vectors
        * activation: list containing all activation vectors
    """

    # Set seed
    tf.random.set_seed(1000)

    # Create model
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer='glorot_uniform',
            input_shape=(
                inputDims,
            )))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer='glorot_uniform'))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train with decreasing learning rate
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01 * (10**(-x)),
                      amsgrad=True),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        model.summary()

        model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=32,
            callbacks=[callback])

    # Save the backup
    model.save(name)

    # Extract weights and biases
    weights = []
    biases = []
    activation = []
    for layer in model.layers:
        weightBias = layer.get_weights()
        if(len(weightBias) == 2):
            weights.append(weightBias[0].T)
            bias = weightBias[1]
            bias = np.reshape(bias, (len(bias), 1))
            biases.append(bias)
        if(len(weightBias) == 1):
            activation.append(weightBias[0])

    return(weights, biases, activation)


def trainBasicClassification(
        hidden,
        inputDims,
        outputDims,
        width,
        cycles,
        epochs,
        alpha,
        patience,
        trainIn,
        trainOut,
        valIn,
        valOut,
        name):
    """ Trains a basic binary classification neural network and returns its
    weights. Uses the amsgrad optimizer and a learning rate of 0.01 which
    decays by a factor of 10 each cycle. The activation function is PReLU.
    Saves the network as name in case something goes wrong with the BNN
    code so the network does not need to be retrained.

    Arguments:
        * hidden: number of hidden layers
        * inputDims: input dimension
        * outputDims: output dimension
        * width: width of hidden layers
        * cycles: number of training cycles with decaying learning rates
        * epochs: number of epochs per cycle
        * alpha: slope value for leaky ReLU
        * patience: early stopping patience
        * trainIn: training input data
        * trainOut: training output data
        * valIn: validation input data
        * valOut: validation output data
    Returns:
        * weights: list containing all weight matrices
        * biases: list containing all bias vectors
        * activation: list containing all activation vectors
    """

    tf.random.set_seed(1000)

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer='glorot_uniform',
            input_shape=(
                inputDims,
            )))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer='glorot_uniform',
            activation='sigmoid'))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)

    for x in range(cycles):

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001 * (10**(-x)),
                      amsgrad=True),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', 'mse'])
        model.summary()
        model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=32,
            callbacks=[callback])

    # Save the backup
    model.save(name)

    # Extract weights and biases
    weights = []
    biases = []
    activation = []
    for layer in model.layers:
        weightBias = layer.get_weights()
        if(len(weightBias) == 2):
            weights.append(weightBias[0].T)
            bias = weightBias[1]
            bias = np.reshape(bias, (len(bias), 1))
            biases.append(bias)
        if(len(weightBias) == 1):
            activation.append(weightBias[0])

    return(weights, biases, activation)
