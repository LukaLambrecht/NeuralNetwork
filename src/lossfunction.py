# -*- coding: utf-8 -*-

################################
# definition of loss functions #
################################

# import external modules
import numpy as np
from abc import ABCMeta,abstractmethod


def get_lossfunction( idstr ):
    ### get a loss function from a given name
    strtof = {}
    strtof['mse'] = MSE()
    strtof['binary_crossentropy'] = BinaryCrossEntropy()
    if idstr in strtof: return strtof[idstr]
    else:
        msg = 'Loss function identifier {} not recognized.'.format(idstr)
        raise Exception(msg)


class LossFunction():
    __metaclass__ = ABCMeta
    ### abstract base class implementation of activation function
    
    def __init__(self):
        ### dummy initializer
        pass
        
    @abstractmethod
    def f(self, labels, predictions):
        ### return function value for given labels and predictions
        if not labels.shape==predictions.shape:
            msg = 'Labels and predictions must have the same shape.'
            raise Exception(msg)
    
    @abstractmethod
    def df(self, labels, predictions):
        ### return derivative for given labels and predictions
        if not labels.shape==predictions.shape:
            msg = 'Labels and predictions must have the same shape.'
            raise Exception(msg)
    
    
class MSE(LossFunction):
    ### mean square error loss function
    
    def __init__(self):
        super(MSE,self).__init__()
        
    def f(self, labels, predictions, domean=True):
        super(MSE,self).f(labels, predictions)
        res = np.square(labels-predictions)
        if domean: res = res.mean()
        return res
    
    def df(self, labels, predictions, domean=True):
        super(MSE,self).df(labels, predictions)
        res = 2*(predictions-labels)
        # note: if prediction > label, df should be positive and vice versa,
        #       so that gradient descent leads in the correct direction.
        if domean: res = res.mean()
        return res
    
    def __str__(self):
        return 'MSE'
    
class BinaryCrossEntropy(LossFunction):
    ### binary cross entropy loss function
    
    def __init__(self):
        super(BinaryCrossEntropy,self).__init__()
        
    def f(self, labels, predictions, domean=True):
        super(BinaryCrossEntropy,self).f(labels, predictions)
        predictions = np.clip(predictions, 1e-5, 1-1e-5)
        term1 = np.multiply(labels, np.log(predictions))
        term2 = np.multiply(1-labels, np.log(1-predictions))
        res = -(term1 + term2)
        if domean: res = res.mean()
        return res
    
    def df(self, labels, predictions, domean=True):
        super(BinaryCrossEntropy,self).df(labels, predictions)
        predictions = np.clip(predictions, 1e-5, 1-1e-5)
        term1 = np.divide(labels, predictions)
        term2 = np.divide(1-labels, 1-predictions)
        res = -(term1 - term2)
        if domean: res = res.mean()
        return res
    
    def __str__(self):
        return 'BinaryCrossEntropy'
