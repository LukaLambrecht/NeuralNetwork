# -*- coding: utf-8 -*-

######################################
# definition of activation functions #
######################################
# note that activation functions always operate on Tensor objects,
# even when these tensors represent vectors or scalars.

import numpy as np
from abc import ABCMeta,abstractmethod
from Tensor import Tensor

def getActivation( idstr ):
	### get an activation function from a given name
	strtof = {}
	strtof['linear'] = LinearActivation()
	strtof['relu'] = ReluActivation()
	strtof['lrelu'] = LReluActivation() # later modify to allow argument passing
	strtof['sigmoid'] = SigmoidActivation()
	if(idstr in strtof): return strtof[idstr]
	else:
		raise Exception('Activation function identifier {} not recognized.'.format(idstr))

class Activation():
	__metaclass__ = ABCMeta
	### abstract base class implementation of activation function
	
	def __init__(self):
		### dummy initializer
		pass
		
	@abstractmethod
	def f(self,arg):
		### return function value at arg
		if not isinstance(arg,Tensor):
			raise Exception('Activation functions can only operate on Tensor objects.')
		return
	
	@abstractmethod
	def df(self,arg):
		### return derivative at arg
		if not isinstance(arg,Tensor):
			raise Exception('Activation functions can only operate on Tensor objects.')
		return	
	
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
		return Tensor(np.where(arg.array>0,arg.array,0.))
	
	def df(self,arg):
		super(ReluActivation,self).df(arg)
		return Tensor(np.where(arg.array>0.,1.,0.))
	
	def __str__(self):
		return 'ReluActivation'
	
class LReluActivation(Activation):
	### leaky relu activation
	
	def __init__(self,negslope=0.1):
		super(LReluActivation,self).__init__()
		self.negslope = negslope
	
	def f(self,arg):
		super(LReluActivation,self).f(arg)
		return Tensor(np.where(arg.array>0,arg.array,self.negslope*arg.array))
	
	def df(self,arg):
		super(LReluActivation,self).df(arg)
		return Tensor(np.where(arg.array>0,1.,self.negslope))
	
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