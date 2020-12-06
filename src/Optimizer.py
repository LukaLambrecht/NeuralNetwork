# -*- coding: utf-8 -*-

##########################################
# implementation of optimizer algorithms #
##########################################

import numpy as np
from abc import ABCMeta,abstractmethod
from Tensor import Tensor

class Optimizer():
	__metaclass__ = ABCMeta
	### abstract base class implementation of optimizer
	
	def __init__(self):
		### dummy initializer
		pass
	
	@abstractmethod
	def update(self,origtensors,tensorgrads):
		### update a given list of tensors with their given gradient
		for origtensor,tensorgrad in zip(origtensors,tensorgrads):
			if not ( isinstance(origtensor,Tensor) and isinstance(tensorgrad,Tensor) ):
				raise Exception('Optimizer got unexpected objects to update (not of Tensor type).')
			if not origtensor.shape==tensorgrad.shape:
				raise Exception('Optimizer got a tensor and its gradient with different shape.')
		return
	
	@abstractmethod
	def checkmag(self,tensors):
		### deal with suspiciously large values in a list of tensors
		for i in range(len(tensors)):
			if(tensors[i].contains_values_above(1e2)):
				tensors[i] = Tensor( np.random.random_sample(tensors[i].shape) )
				print('### WARNING ###: tensor with large values has been reset!')
				print('if this occurs often, this might indicate a problem with normalization or learning rate.')
	
	
class SGD(Optimizer):
	### implementation of stochastic gradient descent optimizer
	
	def __init__(self,learning_rate=0.01,momentum=0.0,nesterov=False):
		super(SGD,self).__init__()
		self.learning_rate = Tensor(np.array([learning_rate]))
		self.momentum = Tensor(np.array([momentum]))
		self.velocities = None
		self.nesterov = nesterov
		
	def update(self,origtensors,tensorgrads):
		super(SGD,self).update(origtensors,tensorgrads)
		ret = []
		if self.velocities is None:
			self.velocities = [Tensor(np.array([0]))*t for t in origtensors]
		for i in range(len(self.velocities)):
			self.velocities[i] += self.momentum * self.velocities[i] - self.learning_rate * tensorgrads[i]
			if not self.nesterov: ret.append( origtensors[i] + self.velocities[i] )
			else: ret.append( origtensors[i] + self.momentum*self.velocities[i] - self.learning_rate*tensorgrads[i] )
		self.checkmag(ret)
		return ret
	
	def __str__(self):
		return 'SGD'
	
class Adam(Optimizer):
	### implementation of adaptive moment estimator
	
	def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-7):
		super(Adam,self).__init__()
		self.learning_rate = Tensor(np.array([learning_rate]))
		self.beta_1 = Tensor(np.array([beta_1]))
		self.onembeta_1 = Tensor(np.array([1-beta_1]))
		self.beta_2 = Tensor(np.array([beta_2]))
		self.onembeta_2 = Tensor(np.array([1-beta_2]))
		self.epsilon = Tensor(np.array([epsilon]))
		self.velocities = None
		self.sqvelocities = None
		self.timecount = 0
		
	def update(self,origtensors,tensorgrads):
		super(Adam,self).update(origtensors,tensorgrads)
		ret = []
		if self.velocities is None:
			self.velocities = [Tensor(np.array([0]))*t for t in origtensors]
			self.sqvelocities = [Tensor(np.array([0]))*t for t in origtensors]
		self.timecount += 1
		for i in range(len(self.velocities)):
			self.velocities[i] = self.beta_1*self.velocities[i] + self.onembeta_1*tensorgrads[i]
			self.sqvelocities[i] = self.beta_2*self.sqvelocities[i] + self.onembeta_2*tensorgrads[i].squared()
			ret.append( origtensors[i] - self.learning_rate*self.velocities[i].divide(self.sqvelocities[i]+self.epsilon) )
		self.checkmag(ret)
		return ret