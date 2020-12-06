# -*- coding: utf-8 -*-

####################################
# testing code for file Network.py #
####################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../src'))
from Tensor import Tensor
from Layer import DenseLayer
from Network import DenseNetwork
from Optimizer import SGD

test_L1_S1_T1 = False
test_L1_S_T1 = False
test_L_S1_T1 = False
test_L_S_T1 = True

### STATUS ###
# - all tests are working!
#   (still quite sensitive to learning rate and input tensor, but no surprise since fully stochastic optimization)
#   tested both with and without bias term in the layers, and all combinations where applicable.

### test network with one layer and input size 1 (i.e. there is only one weight)
### and with one input tensor

if test_L1_S1_T1:
	
	N = DenseNetwork()
	N.add_layer( DenseLayer(1,1,'linear') )
	#N.layers[0].set_weights(Tensor(np.array([[1.]])))
	N.set_loss_function('mse')
	N.set_optimizer( SGD(learning_rate=0.1) )

	intensor = Tensor(np.array([[2]]))
	labels = np.array([1])
	y = N.propagate(intensor)
	print('original network:')
	print(N.layers[0].weights)
	print('original prediction: '+str(y))
	
	backtensors = N.backpropagate(intensor)
	print(backtensors)
	print(backtensors[0])

	for i in range(10):
		N.fit_single_tensor(intensor,labels[0])

	y = N.propagate(intensor)
	print('final prediction '+str(y))

### test network with one layer and input size > 1
### and with one input tensor

if test_L1_S_T1:
	
	N = DenseNetwork()
	N.add_layer( DenseLayer(3,1,'linear') )
	N.set_loss_function('mse')
	N.set_optimizer( SGD(learning_rate=0.01) )
	
	intensor = Tensor(np.array([1,2,-3]))
	labels = np.array([1])
	y = N.propagate(intensor)
	print('original network:')
	print(N.layers[0].weights)
	print('original prediction: '+str(y))
	
	backtensors = N.backpropagate(intensor)
	print(backtensors)
	print(backtensors[0])
	
	for i in range(20):
		N.fit_single_tensor(intensor,labels[0])

	y = N.propagate(intensor)
	print('final prediction '+str(y))
	
### test network with two layers of size 1
###☺ and with one input tensor
	
if test_L_S1_T1:
	
	N = DenseNetwork()
	N.add_layer( DenseLayer(1,1,'linear',biasterm=False ) )
	N.add_layer( DenseLayer(1,1,'linear',biasterm=True) )
	N.set_loss_function('mse')
	N.set_optimizer( SGD(learning_rate=0.05) )
	
	intensor = Tensor(np.array([2]))
	labels = np.array([1])
	y = N.propagate(intensor)
	print('original network:')
	for layer in N.layers: print(layer.weights)
	print('original prediction: '+str(y))
	
	backtensors = N.backpropagate(intensor)
	print(backtensors)
	print(backtensors[0])
	print(backtensors[1])
	
	for i in range(30):
		N.fit_single_tensor(intensor,labels[0])

	y = N.propagate(intensor)
	print('final prediction '+str(y))
	
### test network with multiple layers and input size > 1
### and with one input tensor
	
if test_L_S_T1:
	
	N = DenseNetwork()
	print(N)
	N.add_layer( DenseLayer(3,3,'linear' ) )
	N.add_layer( DenseLayer(3,1,'linear') )
	N.set_loss_function('mse')
	N.set_optimizer( SGD(learning_rate=0.01) )
	N.set_batch_size(1)
	N.set_nepochs(30)
	
	intensor = Tensor(np.array([1,2,-3]))
	labels = np.array([1])
	y = N.propagate(intensor)
	print('original prediction: '+str(y))
	
	#backtensors = N.backpropagate(intensor)
	#print(backtensors)
	#print(backtensors[0])
	
	#for i in range(30):
	#	N.fit_single_tensor(intensor,labels[0])
	
	N.fit(np.array([[1,2,-3]]),labels)

	y = N.propagate(intensor)
	print('final prediction '+str(y))
	
	print(N)