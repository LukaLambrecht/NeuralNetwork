# -*- coding: utf-8 -*-

################################################
# tensor definitions and elementary operations #
################################################

import numpy as np

class Tensor:
	### class representing a 2D tensor, i.e. an array of shape (n1,n2)
	
	array = None
	shape = None
	
	def __init__(self,nparray):
		### constructor from numpy array or scalar
		# allowed shapes are (n) or (n1,n2)
		# in all cases, the resulting Tensor object will be of shape (n1,n2),
		# which means input shape (n) is converted into (n,1)
		if(len(nparray.shape)==2):
			self.array = np.copy(nparray)
			self.shape = self.array.shape
		elif(len(nparray.shape)==1):
			self.array = np.copy(np.expand_dims(nparray,axis=1))
			self.shape = self.array.shape
		else:
			raise Exception('Invalid input shape for Tensor initialisation: {}'.format(nparray.shape))
			
	def __str__(self):
		### string conversion overloading
		return str(self.array)
			
	def __getitem__(self,indices):
		### element access overloading
		# note that python implicitly converts multidim indices to a tuple
		# no index out of bounds checking since implicit in numpy array
		return self.array[indices[0],indices[1]]
		
	def __add__(self,other):
		### addition overloading
		if( self.is_scalar() ):
			return Tensor( self[0,0]+other.array )
		elif( other.is_scalar() ):
			return Tensor( self.array+other[0,0] )
		elif(self.shape!=other.shape):
			raise Exception('Tensors with shapes {} and {} cannot be added.'.format(self.shape,other.shape))
		return Tensor(self.array + other.array)
	
	def __sub__(self,other):
		### subtraction overloading
		return self + Tensor(-other.array)
				
	def __mul__(self,other):
		### multiplication overloading
		# the default behaviour is matrix multiplication, i.e. (n1,m)*(m,n2) -> (n1,n2)
		# if one of the tensors is scalar, element-wise multiplication is performed.
		if( self.is_scalar() ):
			return Tensor( self[0,0]*other.array )
		elif( other.is_scalar() ):
			return Tensor( self.array*other[0,0] )
		elif(self.shape[1]!=other.shape[0]):
			raise Exception('Tensors with shapes {} and {} cannot be multiplied.'.format(self.shape,other.shape))
		return Tensor( np.matmul(self.array,other.array) )
		
	def transpose(self):
		### transpose a tensor
		# the input and output by definition always have a 2D shape
		return Tensor(np.transpose(self.array))
	
	def is_vector(self):
		### check whether tensor has shape (n,1)
		return self.shape[1]==1
	
	def is_scalar(self):
		### check whether tensor has shape (1,1)
		return self.shape==(1,1)
	
	def diag(self):
		### make a diagonal matrix out of a vector
		if not self.is_vector:
			raise Exception('Cannot make a diagonal Tensor out of a Tensor with shape {}'.format(self.shape))
		return Tensor(np.diag(np.squeeze(self.array,axis=1)))
	
	def contains_values_above(self,threshold):
		### return whether a tensor contains values (absolute value) above threshold
		arr = np.abs(self.array)
		return np.any(arr>threshold)
	
	def squared(self):
		### element-wise square
		return Tensor( np.square(self.array) )
	
	def divide(self,other):
		### element-wise division
		return Tensor( np.divide(self.array,other.array) )