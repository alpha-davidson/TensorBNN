import math

import numpy as np
import tensorflow as tf


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
    """ Calculates the log probability of x given x0 and gamma defining
    a cauchy distribution.

    Arguments:
        * gamma: the gamma value for the distribution
        * x0: the mean value for the distribution
        * x: an n-dimensional vectors to have their probabilities calculated
        * dtype: data type of calculation
    Returns:
        * prob: an n-dimensional vector with the log-probabilities of x
    """

    a = tf.math.log(1 + ((x - x0) / gamma)**2)
    b = tf.math.log(tf.cast(math.pi * gamma, dtype))
    c = tf.ones_like(x)
    d = -tf.math.scalar_mul(b, c)
    prob = a + d
    prob = tf.cast(prob, dtype)
    return(prob)


def trainBasicRegression(
        hidden,
        inputDims,
        outputDims,
        width,
        cycles,
        epochs,
        alpha,
        trainIn,
        trainOut,
        valIn,
        valOut,
        name,
        callbacks=True,
        callbackMetric="val_loss",
        patience=10):
    """Trains a basic regression neural network and returns its weights. Uses
    the amsgrad optimizer and a learning rate of 0.01 which decays by a factor
    of 10 each cycle. The activation function is leaky relu with the specified
    alpha value. Saves the network as name in case something goes wrong with
    the BNN code so the network does not need to be retrained.

    Arguments:
        * hidden: number of hidden layers
        * inputDims: input dimension
        * outputDims: output dimension
        * width: width of hidden layers
        * cycles: number of training cycles with decaying learning rates
        * epochs: number of epochs per cycle
        * alpha: slope value for leaky ReLU
        * trainIn: training input data
        * trainOut: training output data
        * valIn: validation input data
        * valOut: validation output data
        * name: name of network
        * callbacks: whether to use callbacks
        * callbackMetric: metric to use for early stopping
        * patience: early stopping patience

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
            kernel_initializer="glorot_uniform",
            input_shape=(
                inputDims,
            )))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform"))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer="glorot_uniform"))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor=callbackMetric, patience=patience, restore_best_weights=True)

    # Train with decreasing learning rate
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01 * (10**(-x)),
                      amsgrad=True),
                      loss="mean_squared_error",
                      metrics=["mean_absolute_error", "mean_squared_error"])
        model.summary()

        if(callbacks):
            model.fit(
                trainIn,
                trainOut,
                validation_data=(
                    valIn,
                    valOut),
                epochs=epochs,
                batch_size=32,
                callbacks=[callback])
        else:
            model.fit(
                trainIn,
                trainOut,
                validation_data=(
                    valIn,
                    valOut),
                epochs=epochs,
                batch_size=32)

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
        trainIn,
        trainOut,
        valIn,
        valOut,
        name,
        callbacks=True,
        callbackMetric="val_loss",
        patience=10):
    """ Trains a basic binary classification neural network and returns its
    weights. Uses the amsgrad optimizer and a learning rate of 0.01 which
    decays by a factor of 10 each cycle. The activation function is leaky_relu
    with the specified alpha value. Saves the network as name in case something
    goes wrong with the BNN code so the network does not need to be retrained.

    Arguments:
        * hidden: number of hidden layers
        * inputDims: input dimension
        * outputDims: output dimension
        * width: width of hidden layers
        * cycles: number of training cycles with decaying learning rates
        * epochs: number of epochs per cycle
        * alpha: slope value for leaky ReLU
        * trainIn: training input data
        * trainOut: training output data
        * valIn: validation input data
        * valOut: validation output data
        * callbacks: whether to use callbacks
        * callbackMetric: metric to use for early stopping
        * patience: early stopping patience
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
            kernel_initializer="glorot_uniform",
            input_shape=(
                inputDims,
            )))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform"))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            outputDims,
            kernel_initializer="glorot_uniform",
            activation="sigmoid"))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor=callbackMetric, patience=patience, restore_best_weights=True)

    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001 * (10**(-x)),
                      amsgrad=True),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy", "mse"])
        model.summary()
        if(callbacks):
            model.fit(
                trainIn,
                trainOut,
                validation_data=(
                    valIn,
                    valOut),
                epochs=epochs,
                batch_size=32,
                callbacks=[callback])
        else:
            model.fit(
                trainIn,
                trainOut,
                validation_data=(
                    valIn,
                    valOut),
                epochs=epochs,
                batch_size=32)

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
