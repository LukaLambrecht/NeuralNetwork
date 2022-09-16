# -*- coding: utf-8 -*-

################################################
# tensor definitions and elementary operations #
################################################

import numpy as np

class Tensor:
  ### class representing a 2D tensor, i.e. an array of shape (n1,n2)

  # class properties
  array = None
  shape = None
	
  def __init__( self, nparray ):
    ### constructor
    # input arguments:
    # - nparray: numpy array for initialization.
    #            note: allowed shapes are (n) or (n1,n2);
    #            in all cases, the resulting Tensor object will be of shape (n1,n2),
    #            which means input shape (n) is converted into (n,1)
    # note: the input array gets deep copied,
    #       so modifying the input array will not modify the Tensor.
    # note: to check if removing the copy can give speedup.
    if(len(nparray.shape)==2):
      self.array = np.copy(nparray)
      self.shape = self.array.shape
    elif(len(nparray.shape)==1):
      self.array = np.copy(np.expand_dims(nparray,axis=1))
      self.shape = self.array.shape
    else:
      raise Exception('Invalid input shape for Tensor initialisation: {}'.format(nparray.shape))
			
  def __str__( self ):
    ### string conversion overloading
    return str(self.array)
			
  def __getitem__( self, indices ):
    ### element access overloading
    # note: python implicitly converts multidim indices to a tuple,
    #       so no index out of bounds checking needed here,
    #       since implicit in numpy array
    return self.array[indices[0],indices[1]]

  def add_bias( self, bias=1, inplace=False ):
    ### add a bias term to a vector tensor
    if not self.is_vector():
      raise Exception('Cannot add a bias term to a tensor of shape {}'.format(self.shape))
    if not inplace: return Tensor(np.append(self.array,np.ones((1,1))*bias))
    else: self.array = np.append(self.array,np.ones((1,1))*bias)

  def remove_bias( self, inplace=False ):
    ### remove a bias terms (assumed to be in the last colunn of a 2D tensor)
    if not inplace: return Tensor(self.array[:,:-1])
    else: self.array = self.array[:,:-1]
		
  def __add__( self, other ):
    ### addition overloading
    # note: a new tensor is created that is the sum of both,
    #       leaving the original two tensors unmodified.
    if( self.is_scalar() ):
      return Tensor( self[0,0]+other.array )
    elif( other.is_scalar() ):
      return Tensor( self.array+other[0,0] )
    elif(self.shape!=other.shape):
      raise Exception('Tensors with shapes {} and {} cannot be added.'.format(self.shape,other.shape))
    return Tensor(self.array + other.array)

  def increase_by( self, other ):
    ### add another tensor to the current one
    # note: the original tensor is modified in-place.
    if( self.is_scalar() ): 
      self.array = self[0,0] + other.array
      self.shape = self.array.shape
    elif( other.is_scalar() ):
      self.array = self.array + other[0,0]
    elif(self.shape!=other.shape):
      raise Exception('Tensors with shapes {} and {} cannot be added.'.format(self.shape,other.shape))
    self.array += other.array

  def __neg__( self ):
    ### unary minus sign overloading
    # note: a new tensor is created that is the negative of the original one,
    #       leaving the original tensor unmodified.
    return Tensor(-self.array)

  def negate( self ):
    ### negate the current tensor
    # note: the original tensor is modified in-place.
    self.array = -self.array

  def __sub__( self, other ):
    ### subtraction overloading
    # note: a new tensor is created that is the subtraction of both,
    #       leaving the original two tensors unmodified.
    return self + (-other)

  def decrease_by( self, other ):
    ### subtract another tensor from the current one
    # note: the original tensor is modified in-place.
    if( self.is_scalar() ):
      self.array = self[0,0] - other.array
      self.shape = self.array.shape
    elif( other.is_scalar() ):
      self.array = self.array - other[0,0]
    elif(self.shape!=other.shape):
      raise Exception('Tensors with shapes {} and {} cannot be subtracted.'.format(self.shape,other.shape))
    self.array -= other.array
				
  def __mul__( self, other ):
    ### multiplication overloading
    # note: the default behaviour is matrix multiplication, i.e. (n1,m)*(m,n2) -> (n1,n2)
    #       if one of the tensors is scalar, element-wise multiplication is performed.
    # note: a new tensor is created that is the multiplication of both,
    #       leaving the original two tensors unmodified.
    if( self.is_scalar() ):
      return Tensor( self[0,0]*other.array )
    elif( other.is_scalar() ):
      return Tensor( self.array*other[0,0] )
    elif(self.shape[1]!=other.shape[0]):
      raise Exception('Tensors with shapes {} and {} cannot be multiplied.'.format(self.shape,other.shape))
    return Tensor( np.matmul(self.array,other.array) )

  def rightmultiply_by( self, other ):
    ### multiply this tensor with another one.
    # schematically: this = this * other.
    # note: the default behaviour is matrix multiplication, i.e. (n1,m)*(m,n2) -> (n1,n2)
    #       if one of the tensors is scalar, element-wise multiplication is performed.
    # note: the original tensor is modified in-place.
    if( self.is_scalar() ):
      self.array = self[0,0]*other.array
      self.shape = self.array.shape
    elif( other.is_scalar() ):
      self.array *= other[0,0]
    elif(self.shape[1]!=other.shape[0]):
      raise Exception('Tensors with shapes {} and {} cannot be multiplied.'.format(self.shape,other.shape))
    self.array = np.matmul(self.array, other.array)
    self.shape = self.array.shape

  def leftmultiply_by( self, other ):
    ### multiply this tensor with another one.
    # schematically: this = other * this.
    # note: the default behaviour is matrix multiplication, i.e. (n1,m)*(m,n2) -> (n1,n2)
    #       if one of the tensors is scalar, element-wise multiplication is performed.
    # note: the original tensor is modified in-place.
    if( self.is_scalar() ):
      self.array = self[0,0]*other.array
      self.shape = self.array.shape
    elif( other.is_scalar() ):
      self.array *= other[0,0]
    elif(other.shape[1]!=self.shape[0]):
      raise Exception('Tensors with shapes {} and {} cannot be multiplied.'.format(other.shape,self.shape))
    self.array = np.matmul(other.array, self.array)
    self.shape = self.array.shape
		
  def transpose( self, inplace=False ):
    ### transpose a tensor
    # the input and output by definition always have a 2D shape
    if not inplace: return Tensor(np.transpose(self.array))
    else: self.array = np.transpose(self.array)
	
  def is_vector( self ):
    ### check whether tensor has shape (n,1)
    return self.shape[1]==1
	
  def is_scalar( self ):
    ### check whether tensor has shape (1,1)
    return self.shape==(1,1)
	
  def diag( self, inplace=False ):
    ### make a diagonal matrix out of a vector
    if not self.is_vector():
      raise Exception('Cannot make a diagonal Tensor out of a Tensor with shape {}'.format(self.shape))
    if not inplace: return Tensor(np.diag(np.squeeze(self.array,axis=1)))
    else:
      self.array = np.diag(np.squeeze(self.array,axis=1))
      self.shape = self.array.shape
	
  def contains_values_above(self,threshold):
    ### return whether a tensor contains values (absolute value) above threshold
    arr = np.abs(self.array)
    return np.any(arr>threshold)
	
  def squared( self, inplace=False ):
    ### element-wise square
    if not inplace: return Tensor( np.square(self.array) )
    else: self.array = np.square(self.array)
	
  def divide( self, other ):
    ### element-wise division
    # note: a new tensor is created that is the multiplication of both,
    #       leaving the original two tensors unmodified.
    return Tensor( np.divide(self.array,other.array) )

  def divide_by( self, other ):
    ### divide the elements of the current tensor by another one
    # note: the original tensor is modified in-place.
    self.array = np.divide(self.array,other.array)
