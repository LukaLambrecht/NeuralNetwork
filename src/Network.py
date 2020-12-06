# -*- coding: utf-8 -*-

#########################
# definition of network #
#########################

import sys
import os
import numpy as np
from Tensor import Tensor
from Layer import DenseLayer
from Optimizer import Optimizer
from LossFunction import getLossFunction
sys.path.append(os.path.abspath('../diag'))
from NetworkHistory import NetworkHistory
sys.path.append(os.path.abspath('../preprocessing'))
import preprocessing as prpr

class DenseNetwork():
	### an ordinary dense feed-forward neural network consisting of DenseLayers
	
	def __init__(self):
		self.layers = []
		self.input_dim = None
		self.optimizer = None
		self.batch_size = None
		self.nepochs = None
		self.loss_function = None
		self.metrics = []
		self.history = NetworkHistory()
		
	def __str__(self):
		### string conversion overloading
		ninfo = '--- DenseNetwork ---\n'
		ninfo += 'general info: optimizer = {},'.format(self.optimizer)
		ninfo += ' batch size = {},'.format(self.batch_size)
		ninfo += ' number of epochs = {},'.format(self.nepochs)
		ninfo += ' loss function = {}\n'.format(self.loss_function)
		ninfo += 'number of layers: {}'.format(len(self.layers))
		if len(self.layers)==0: return ninfo
		ninfo += '\n'+'-'*30+'\n'
		ninfo += 'input dimension: {}\n'.format(self.input_dim)
		for l in self.layers[::-1]: ninfo += str(l)+'\n'
		ninfo += '-'*30+'\n'
		return ninfo
	
	def add_layer(self,layer):
		### add a layer to a DenseNetwork
		# note that the layer is inserted at the front,
		# i.e. the output layer is at index 0, the input layer at index -1
		if not isinstance(layer,DenseLayer):
			raise Exception('Only DenseLayer objects can be added to this DenseNetwork.')
		if( len(self.layers)>0 and layer.input_dim!=self.layers[0].output_dim ):
			raise Exception('Input and output dimensions of successive layers do not match.')
		self.layers.insert(0,layer)
		if(len(self.layers)==1): self.input_dim = layer.input_dim
		
	def set_optimizer(self,optimizer):
		### set the optimizer for this network
		# type of input argument must be 'Optimizer'
		if not isinstance(optimizer,Optimizer):
			raise Exception('Given optimizer is not of type Optimizer.')
		self.optimizer = optimizer
		
	def set_nepochs(self,nepochs):
		### set the number of epochs for this network
		# input is assumed to be an integer
		self.nepochs = nepochs
		
	def set_batch_size(self,batch_size):
		### set the batch size for this network
		# input is assumed to be an integer
		self.batch_size = batch_size
		
	def set_loss_function(self,loss_function):
		### set the loss function for this network
		# input type is assumed to be str (loss function identifier)
		self.loss_function = getLossFunction(loss_function)
		self.metrics.append(self.loss_function)
		
	def add_metric(self,metric):
		### add a metric that will be used for intermediate evaluation
		# input type is assumed to be str (metric identifier)
		# todo: consistently implement metrics and loss functions
		pass
		
	def propagate(self,intensor,return_ext=False):
		### propagate a given input tensor through the current network
		# in the default case, only the final output tensor is returned
		# if return_ext is True, all intermediate outputs are returned as well
		if not (intensor.is_vector() and intensor.shape[0]==self.input_dim):
			raise Exception('Tensor with shape {} is invalid input for DenseNetwork with input dimension {}'
									 .format(intensor.shape,self.input_dim))
		outtensors = []
		for layer in self.layers[-1::-1]:
			outtensors.insert( 0,layer.propagate(intensor) )
			intensor = outtensors[0]
		if not return_ext: return outtensors[0]
		else: return outtensors
	
	def backpropagate(self,intensor,intermediate_outputs=None):
		### calculate the gradient of the current network prediction
		# the intermediate_outputs argument allows to pass the intermediate outputs
		# if this is None, they are recalculated in a forward pass
		if intermediate_outputs is None:
			intermediate_outputs = self.propagate(intensor,return_ext=True)
		intermediate_outputs.append(intensor)
		backtensors = []
		back_tensor = Tensor(np.array([1]))
		for i,layer in enumerate(self.layers):
			(closing_tensor,mult) = layer.backpropagate(intermediate_outputs[i+1])
			closing_input = intermediate_outputs[i+1]
			if layer.biasterm: closing_input = Tensor(np.append(closing_input.array,np.array([[1]])))
			backtensors.append( (closing_input*back_tensor*closing_tensor).transpose() )
			back_tensor = back_tensor * mult
		return backtensors
	
	def fit_single_tensor(self,intensor,label):
		### update the current network given a single input tensor with label
		# for internal and debugging use only, use externally only within the more general 'fit' method
		outtensors = self.propagate(intensor,return_ext=True)
		backtensors = self.backpropagate(intensor,intermediate_outputs=outtensors)
		lossgrad = self.loss_function.df(np.array([label]),np.squeeze(outtensors[0].array,axis=1))
		backtensors = [ Tensor(np.array([lossgrad]))*t for t in backtensors ]
		origtensors = [ l.weights for l in self.layers ]
		newtensors = self.optimizer.update(origtensors,backtensors)
		for layer,newtensor in zip(self.layers,newtensors):
			layer.set_weights( newtensor )
			
	def fit_batch(self,intensors,labels):
		### update the current network given a batch of training tensors with labels
		batch_size = len(intensors)
		# initialization using first tensor
		outtensors = self.propagate(intensors[0],return_ext=True)
		lossgrad = self.loss_function.df(np.array([labels[0]]),np.squeeze(outtensors[0].array,axis=1))
		backtensors = self.backpropagate(intensors[0],intermediate_outputs=outtensors)
		backtensors = [ Tensor(np.array([lossgrad]))*t for t in backtensors ]
		# sum results for all subsequent tensors
		for intensor,label in zip(intensors[1:],labels[1:]):
			outtensors = self.propagate(intensor,return_ext=True)
			lossgrad = self.loss_function.df(np.array([label]),np.squeeze(outtensors[0].array,axis=1))
			backtensors_i = self.backpropagate(intensor,intermediate_outputs=outtensors)
			backtensors_i = [ Tensor(np.array([lossgrad]))*t for t in backtensors_i ]
			backtensors = [t1+t2 for t1,t2 in zip(backtensors,backtensors_i)]
		# average
		backtensors = [ Tensor(np.array([1./batch_size]))*t for t in backtensors]
		# now update
		origtensors = [ l.weights for l in self.layers ]
		newtensors = self.optimizer.update(origtensors,backtensors)
		for layer,newtensor in zip(self.layers,newtensors):
			layer.set_weights( newtensor )
	
	def fit(self,X_train,labels,validation_fraction=0.1):
		### fit the current network to inputs with given labels
		# input arguments:
		# - X_train is a numpy array of shape (ninstances,nfeatures)
		# - labels is a numpy array of shape (ninstances)
		
		# check if all conditions are met for sensible training
		if( len(X_train.shape)!=2 ): raise Exception('X_train has unexpected shape: {}'.format(X_train.shape))
		if( len(labels.shape)!=1 ): raise Exception('Labels have unexpected shape: {}'.format(labels.shape))
		if( X_train.shape[0]!=len(labels) ): raise Exception('Number of instances in X_train and labels must be equal.')
		if( len(self.layers)==0 ): raise Exception('This network has no layers yet.')
		if( X_train.shape[1]!=self.input_dim ): raise Exception('X_train has {} features, but network has input size {}'
																																							.format(X_train.shape[1],self.input_dim))
		if( self.nepochs==None ): raise Exception('Number of epochs was not yet set for this network.')
		if( self.optimizer==None ): raise Exception('Optimizer was not yet set for this network.')
		if( self.batch_size==None ): raise Exception('Batch size was not yet set for this network.')
		if( self.loss_function==None ): raise Exception('Loss function was not yet set for this network.')
		
		# define training/validation sets
		ninstances = X_train.shape[0]
		ntrain = ninstances
		do_validation = True
		splitindex = int(validation_fraction*ninstances)
		if splitindex == 0: do_validation = False
		if do_validation:
			(X_train,labels) = prpr.shuffle_training_set(X_train,labels)
			X_val = X_train[:splitindex,:]
			labels_val = labels[:splitindex]
			X_train = X_train[splitindex:,:]
			labels = labels[splitindex:]
			ntrain = ninstances-splitindex
		
		print('--- starting network training ---')
		print('shape of training set: {}'.format(X_train.shape))
		if do_validation: print('shape of validation set: {}'.format(X_val.shape))
		print('number of epochs: {}'.format(self.nepochs))
		nbatches = int(ntrain/self.batch_size)
		
		# loop over epochs
		for epoch in range(self.nepochs):
			print('epoch {}/{}: batch {}/{}'.format(epoch+1,self.nepochs,0,nbatches),end='')
			# make and submit batches
			startindex = 0
			batchn = 1
			while startindex + 2*self.batch_size <= ntrain:
				print('\r'+'epoch {}/{}: batch {}/{}'.format(epoch+1,self.nepochs,batchn,nbatches),end='')
				intensors_batch = [ Tensor(thisarray) for thisarray in X_train[startindex:startindex+self.batch_size] ]
				labels_batch = labels[startindex:startindex+self.batch_size]
				self.fit_batch(intensors_batch,labels_batch)
				if do_validation: self.history.add_entry_info( epoch=epoch+1, batch=batchn, metrics=self.evaluate_metrics(X_val, labels_val) )
				startindex += self.batch_size
				batchn += 1
			print('\r'+'epoch {}/{}: batch {}/{}'.format(epoch+1,self.nepochs,batchn,nbatches))
			X_train_batch = [ Tensor(thisarray) for thisarray in X_train[startindex:] ]
			labels_batch = labels[startindex:]
			self.fit_batch(X_train_batch,labels_batch)
			if do_validation: self.history.add_entry_info( epoch=epoch+1, batch=batchn, metrics=self.evaluate_metrics(X_val, labels_val) )
		print('--- finished network training ---')
					
	def predict(self,X_test):
		### calculate the network output for given instances
		# X_test is a numpy array of shape (ninstances,nfeatures)
		
		# check if all conditions are met
		if( len(X_test.shape)!=2 ): raise Exception('X_test has unexpected shape: {}'.format(X_test.shape))
		if( len(self.layers)==0 ): raise Exception('This network has no layers yet.')
		if( X_test.shape[1]!=self.input_dim ): raise Exception('X_train has {} features, but network has input size {}'
																																							.format(X_test.shape[1],self.input_dim))
		
		# calculate the network output
		pred = np.zeros(len(X_test))
		for i in range(X_test.shape[0]):
			pred[i] = np.squeeze(self.propagate(Tensor(X_test[i,:])).array,axis=1)
		return pred
	
	def evaluate_metrics(self,X_test,labels):
		### evaluate the internally stored metric functions on a given test set
		# note: mostly for internal usage to store training history
		# note: 'metrics' are here taken equivalent to loss function!
		#       still to decide how to treat loss functions and metrics uniformly
		res = {}
		pred = self.predict(X_test)
		for metric in self.metrics:
			res[metric] = metric.f(labels,pred)
		return res
	
	def plot_weights(self):
		### plot all weight matrices for this network
		layern = 1
		for l in self.layers[::-1]:
			l.plot_weights(title='weights for layer {}'.format(layern))
			layern += 1