# -*- coding: utf-8 -*-

################################
# definition of loss functions #
################################

import numpy as np
from abc import ABCMeta,abstractmethod

def getLossFunction( idstr ):
	### get a loss function from a given name
	strtof = {}
	strtof['mse'] = MSE()
	strtof['binary_crossentropy'] = BinaryCrossEntropy()
	if(idstr in strtof): return strtof[idstr]
	else:
		raise Exception('Loss function identifier {} not recognized.'.format(idstr))

class LossFunction():
	__metaclass__ = ABCMeta
	### abstract base class implementation of activation function
	
	def __init__(self):
		### dummy initializer
		pass
		
	@abstractmethod
	def f(self,labels,predictions):
		### return function value for given labels and predictions
		if not labels.shape==predictions.shape:
			raise Exception('Labels and predictions must have the same shape.')
		return
	
	@abstractmethod
	def df(self,labels,predictions):
		### return derivative for given labels and predictions
		if not labels.shape==predictions.shape:
			raise Exception('Labels and predictions must have the same shape.')
		return
	
	
class MSE(LossFunction):
	### mean square error loss function
	
	def __init__(self):
		super(MSE,self).__init__()
		
	def f(self,labels,predictions):
		super(MSE,self).f(labels,predictions)
		return np.square(labels-predictions).mean()
	
	def df(self,labels,predictions):
		super(MSE,self).df(labels,predictions)
		return 2*(predictions-labels).mean()
	
	def __str__(self):
		return 'MSE'
	
class BinaryCrossEntropy(LossFunction):
	### binary cross entropy loss function
	
	def __init__(self):
		super(BinaryCrossEntropy,self).__init__()
		
	def f(self,labels,predictions):
		super(BinaryCrossEntropy,self).f(labels,predictions)
		return (np.multiply(labels,predictions)+np.multiply(1-labels,1-predictions)).mean()
	
	def df(self,labels,predictions):
		super(BinaryCrossEntropy,self).df(labels,predictions)
		return np.nan_to_num(np.divide( (labels-predictions),np.multiply(predictions,1-predictions) )).mean()
	
	def __str__(self):
		return 'BinaryCrossEntropy'