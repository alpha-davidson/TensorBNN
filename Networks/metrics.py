import tensorflow as tf


class Metric(object):
    """ A basic metric object. This can be implemented into any desired
    metric within the BNN training loop.
    """
    def __init__(self, scaleExp = False, mean=0, sd=1,  *argv, **kwargs):
        self.scaleExp = scaleExp
        self.mean = mean
        self.sd = sd
    
    def calculate(self, predictionsTrain, predictionValidate, realTrain,
                      realValidate, *argv, **kwargs):
        """ Calculates the metric

        Arguments:
            * predictionsTrain: training predictions
            * predictionsValidate: validation predictions
            * realTrain: real training values
            * realValidate: real validation values
        """
        pass
    
    def display(self):
        """Displays the metric"""
        pass
        

class SquaredError(Metric):
    """ Calculates the mean squared error of a prediction.
    """
    
    def calculate(self,  predictionsTrain, predictionsValidate, realTrain, 
                  realValidate):
        
        predictionsTrain = tf.add(tf.multiply(tf.transpose(predictionsTrain),
                                              self.sd), self.mean)
        predictionsValidate = tf.add(tf.multiply(tf.transpose(predictionsValidate), 
                                                 self.sd), self.mean)
        
        realTrain = tf.add(tf.multiply(realTrain, self.sd), self.mean)
        realValidate = tf.add(tf.multiply(realValidate, self.sd), self.mean)

        if(self.scaleExp):
            predictionsTrain = tf.exp(predictionsTrain)
            realTrain = tf.exp(realTrain)
            realValidate = tf.exp(realValidate)

        realTrain = tf.reshape(realTrain, predictionsTrain.shape)
        realValidate = tf.reshape(realValidate, predictionsValidate.shape)
        
        
        squaredError = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(
                predictionsTrain, realTrain))
        self.squaredErrorTrain=squaredError.numpy()
    
        squaredError = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(
                predictionsValidate, realValidate))
        self.squaredErrorValidate=squaredError.numpy()
    
    def display(self):
        
        print("training squared error{: 9.5f}".format(self.squaredErrorTrain),
                    "validation squared error{: 9.5f}".format(
                        self.squaredErrorValidate))

class PercentError(Metric):
    """Calculates percent error of a prediction"""
    def __init__(self, scaleExp = False, mean=0, sd=1, *argv, **kwargs):
        self.scaleExp =scaleExp
        self.mean = mean
        self.sd = sd
          
        
    def calculate(self,  predictionsTrain, predictionsValidate, realTrain,
                  realValidate):
        predictionsTrain = tf.add(tf.multiply(tf.transpose(predictionsTrain), 
                                              self.sd), self.mean)
        predictionsValidate = tf.add(tf.multiply(tf.transpose(predictionsValidate), 
                                                 self.sd), self.mean)
        
        realTrain = tf.add(tf.multiply(realTrain, self.sd), self.mean)
        realValidate = tf.add(tf.multiply(realValidate, self.sd), self.mean)

        if(self.scaleExp):
            predictionsTrain = tf.exp(predictionsTrain)
            predictionsValidate = tf.exp(predictionsValidate)
            realTrain = tf.exp(realTrain)
            realValidate = tf.exp(realValidate)

        realTrain = tf.reshape(realTrain, predictionsTrain.shape)
        realValidate = tf.reshape(realValidate, predictionsValidate.shape)
                
        self.percentErrorTrain = tf.reduce_mean(
            input_tensor=tf.multiply(
                tf.abs(tf.divide(tf.subtract(predictionsTrain, realTrain),
                                 realTrain)), 100))
        self.percentErrorValidate = tf.reduce_mean(
            input_tensor=tf.multiply(
                tf.abs(tf.divide(tf.subtract(predictionsValidate, realValidate), 
                                 realValidate)), 100))
   
    def display(self):
        print("training percent error{: 7.3f}".format(self.percentErrorTrain),
                      "training percent error{: 7.3f}".format(self.percentErrorValidate))
        
class Accuracy(Metric):
    """ Caluclates the accuracy of predictions """
    def calculate(self,  predictionsTrain, predictionsValidate, realTrain, 
                  realValidate):
        predictionsTrain = tf.add(tf.multiply(tf.transpose(predictionsTrain), 
                                              self.sd), 
                        self.mean)
        predictionsValidate = tf.add(tf.multiply(tf.transpose(predictionsValidate), 
                                                 self.sd), 
                        self.mean)
        realTrain = tf.add(tf.multiply(realTrain, self.sd), self.mean)
        realValidate = tf.add(tf.multiply(realValidate, self.sd), self.mean)

        if(self.scaleExp):
            predictionsTrain = tf.exp(predictionsTrain)
            predictionsValidate = tf.exp(predictionsValidate)
            realTrain = tf.exp(realTrain)
            realValidate = tf.exp(realValidate)
        
        realTrain = tf.reshape(realTrain, predictionsTrain.shape)
        realValidate = tf.reshape(realValidate, predictionsValidate.shape)

        self.accuracyTrain = 1 - tf.reduce_mean(tf.abs(
                                            realTrain - tf.round(predictionsTrain)))
        self.accuracyValidate = 1 - tf.reduce_mean(tf.abs(
                                            realValidate - tf.round(predictionsValidate)))

 
    def display(self):
        print("training accuracy{: 9.5f}".format(self.accuracyTrain),
                      "validation accuracy{: 9.5f}".format(
                          self.accuracyValidate))
        