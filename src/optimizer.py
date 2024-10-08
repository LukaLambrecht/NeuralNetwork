# -*- coding: utf-8 -*-

##########################################
# implementation of optimizer algorithms #
##########################################

# import external modules
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

# import local modules
from tensor import Tensor


class Optimizer():
    __metaclass__ = ABCMeta
    ### abstract base class implementation of optimizer
    
    def __init__(self):
        ### dummy initializer
        pass
    
    @abstractmethod
    def update(self, origtensors, tensorgrads):
        ### update a given list of tensors with their given gradient
        for origtensor,tensorgrad in zip(origtensors,tensorgrads):
            if not ( isinstance(origtensor,Tensor) and isinstance(tensorgrad,Tensor) ):
                raise Exception('Optimizer got unexpected objects to update (not of Tensor type).')
            if not origtensor.shape==tensorgrad.shape:
                raise Exception('Optimizer got a tensor and its gradient with different shape.')
    
    @abstractmethod
    def checkmag(self, tensors):
        ### deal with suspiciously large values in a list of tensors
        for i in range(len(tensors)):
            if(tensors[i].contains_values_above(1e2)):
                tensors[i] = Tensor( np.random.random_sample(tensors[i].shape) )
                msg = '### WARNING ###: tensor with large values has been reset!'
                msg += ' If this occurs often, this might indicate a problem'
                msg += ' with normalization or learning rate.'
                print(msg)
    
    
class SGD(Optimizer):
    ### implementation of stochastic gradient descent optimizer
    
    def __init__(self,
            learning_rate=0.1,
            momentum=0.0,
            nesterov=False,
            gradclip=None):
        super(SGD,self).__init__()
        self.learning_rate = Tensor(np.array([learning_rate]))
        self.momentum = Tensor(np.array([momentum]))
        self.velocities = None
        self.nesterov = nesterov
        self.gradclip = gradclip
        
    def update(self, origtensors, tensorgrads):
        super(SGD,self).update(origtensors,tensorgrads)
        # clip gradients if required
        if self.gradclip is not None:
            tensorgrads = [t.clip(-self.gradclip, self.gradclip) for t in tensorgrads]
        # initialize result
        res = []
        # initialize velocities at zero
        if self.velocities is None:
            self.velocities = [Tensor(np.array([0]))*t for t in origtensors]
        for i in range(len(origtensors)):
            # update velocities
            self.velocities[i] = self.momentum * self.velocities[i] + self.learning_rate * tensorgrads[i]
            # todo: implement nesterov
            res.append( origtensors[i] - self.velocities[i] )
        self.checkmag(res)
        return res
    
    def __str__(self):
        return 'SGD'


class RMSprop(Optimizer):
    ### implementation of RMSprop optimizer

    def __init__(self,
            learning_rate=0.1,
            beta=0.9,
            epsilon=1e-8):
        super(RMSprop,self).__init__()
        self.learning_rate = Tensor(np.array([learning_rate]))
        self.beta = Tensor(np.array([beta]))
        self.onembeta = Tensor(np.array([1-beta]))
        self.epsilon = Tensor(np.array([epsilon]))
        self.sqgrads = None

    def update(self, origtensors, tensorgrads):
        super(RMSprop,self).update(origtensors,tensorgrads)
        # initialize result
        res = []
        # initialize squared gradients at zero
        if self.sqgrads is None:
            self.sqgrads = [Tensor(np.array([0]))*t for t in origtensors]
        for i in range(len(origtensors)):
            # update squared gradients
            self.sqgrads[i] = self.beta * self.sqgrads[i] + self.onembeta * tensorgrads[i].squared()
            step = -self.learning_rate * tensorgrads[i].divide(self.sqgrads[i].sqrt() + self.epsilon)
            res.append( origtensors[i] + step )
        self.checkmag(res)
        return res

    def __str__(self):
        return 'RMSprop'

    
class Adam(Optimizer):
    ### implementation of adaptive moment estimator
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super(Adam,self).__init__()
        self.learning_rate = Tensor(np.array([learning_rate]))
        self.beta_1 = Tensor(np.array([beta_1]))
        self.onembeta_1 = Tensor(np.array([1-beta_1]))
        self.beta_2 = Tensor(np.array([beta_2]))
        self.onembeta_2 = Tensor(np.array([1-beta_2]))
        self.epsilon = Tensor(np.array([epsilon]))
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, origtensors, tensorgrads):
        super(Adam,self).update(origtensors,tensorgrads)
        res = []
        if self.m is None or self.v is None:
            self.m = [Tensor(np.array([0]))*t for t in origtensors]
            self.v = [Tensor(np.array([0]))*t for t in origtensors]
        self.t += 1
        for i in range(len(origtensors)):
            self.m[i] = self.beta_1*self.m[i] + self.onembeta_1*tensorgrads[i]
            self.v[i] = self.beta_2*self.v[i] + self.onembeta_2*(tensorgrads[i].squared())
            mhat = self.m[i] * Tensor( 1./(1 - np.power(self.beta_1.array, self.t)) )
            vhat = self.v[i] * Tensor( 1./(1 - np.power(self.beta_2.array, self.t)) )
            res.append( origtensors[i] - self.learning_rate * mhat.divide( vhat.sqrt() + self.epsilon) )
        self.checkmag(res)
        return res
