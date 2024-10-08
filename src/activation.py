# -*- coding: utf-8 -*-

######################################
# definition of activation functions #
######################################

# Note: the activation functions defined here always operate on Tensor objects,
#       even when these tensors represent vectors or scalars.
# Note: the return object of each activation function is a new Tensor object.

# import external modules
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

# import local modules
from tensor import Tensor


def get_activation( idstr ):
    ### get an activation function from a given name
    strtof = {}
    strtof['linear'] = LinearActivation()
    strtof['relu'] = ReluActivation()
    strtof['lrelu'] = LReluActivation()
    strtof['sigmoid'] = SigmoidActivation()
    strtof['tanh'] = TanhActivation()
    if idstr in strtof: return strtof[idstr]
    else:
        msg = 'Activation function identifier {} not recognized.'.format(idstr)
        raise Exception(msg)


class Activation():
    __metaclass__ = ABCMeta
    ### abstract base class implementation of activation function
    
    def __init__(self):
        ### dummy initializer
        pass
        
    @abstractmethod
    def f(self, arg):
        ### return function value at arg
        if not isinstance(arg, Tensor):
            msg = 'Activation functions can only operate on Tensor objects.'
            raise Exception(msg)
    
    @abstractmethod
    def df(self,arg):
        ### return derivative at arg
        if not isinstance(arg, Tensor):
            msg = 'Activation functions can only operate on Tensor objects.'
            raise Exception(msg)
    

class LinearActivation(Activation):
    ### linear activation function
    
    def __init__(self):
        super(LinearActivation,self).__init__()
        
    def f(self,arg):
        super(LinearActivation,self).f(arg)
        return Tensor(arg.array)
    
    def df(self,arg):
        super(LinearActivation,self).df(arg)
        return Tensor(np.ones(arg.shape))
    
    def __str__(self):
        return 'LinearActivation'


class ReluActivation(Activation):
    ### rectified linear unit activation
    
    def __init__(self):
        super(ReluActivation,self).__init__()
        
    def f(self,arg):
        super(ReluActivation,self).f(arg)
        return Tensor( np.where(arg.array>0, arg.array, 0.) )
    
    def df(self,arg):
        super(ReluActivation,self).df(arg)
        return Tensor( np.where(arg.array>0., 1., 0.) )
    
    def __str__(self):
        return 'ReluActivation'
    
class LReluActivation(Activation):
    ### leaky relu activation
    
    def __init__(self,negslope=0.1):
        super(LReluActivation,self).__init__()
        self.negslope = negslope
    
    def f(self,arg):
        super(LReluActivation,self).f(arg)
        return Tensor( np.where(arg.array>0, arg.array, self.negslope*arg.array) )
    
    def df(self,arg):
        super(LReluActivation,self).df(arg)
        return Tensor( np.where(arg.array>0,1., self.negslope) )
    
    def __str__(self):
        return 'LReluActivation (neg. slope = {})'.format(self.negslope)
    
class SigmoidActivation(Activation):
    ### sigmoid function activation
    
    def __init__(self):
        super(SigmoidActivation,self).__init__()
        
    def f(self,arg):
        super(SigmoidActivation,self).f(arg)
        return Tensor( np.divide(1,1+np.exp(-arg.array)) )
    
    def df(self,arg):
        super(SigmoidActivation,self).df(arg)
        return Tensor( np.nan_to_num(np.divide(np.exp(arg.array),np.power(1+np.exp(arg.array),2))) )
    
    def __str__(self):
        return 'SigmoidActivation'

class TanhActivation(Activation):
    ### tanh function activation

    def __init__(self):
        super(TanhActivation,self).__init__()

    def f(self,arg):
        super(TanhActivation,self).f(arg)
        exp = np.exp(arg.array)
        nexp = np.exp(-arg.array)
        return Tensor( np.divide(exp - nexp, exp + nexp) )

    def df(self,arg):
        super(TanhActivation,self).df(arg)
        exp = np.exp(arg.array)
        nexp = np.exp(-arg.array)
        return Tensor( np.nan_to_num( 4 / np.power(exp + nexp, 2) ) )

    def __str__(self):
        return 'TanhActivation'
