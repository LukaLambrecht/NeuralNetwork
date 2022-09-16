# -*- coding: utf-8 -*-

########################################################
# definition of layers and (back)propagation functions #
########################################################

import numpy as np
import matplotlib.pyplot as plt
from Tensor import Tensor
from Activation import getActivation

class DenseLayer():
  ### a dense layer for an ordinary neural network
	
  # class properties
  weights = None
  biasterm = None
  input_dim = None
  output_dim = None
  activation = None
	
  def __init__( self, input_dim, output_dim,
                activation='linear', biasterm=True ):
    ### constructor from given input and output dimension.
    # note: the underlying tensor is initialized randomly
    if biasterm: 
      initarray = np.random.rand(output_dim,input_dim+1)
      self.biasterm = True
    else: 
      initarray = np.random.rand(output_dim,input_dim)
      self.biasterm = False
    self.weights = Tensor(initarray)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.activation = getActivation(activation)
		
  def __str__( self ):
    ### string conversion overloading
    linfo = 'DenseLayer:'
    linfo += ' input dim = {},'.format(self.input_dim)
    linfo += ' output dim = {},'.format(self.output_dim)
    linfo += ' weight shape = {},'.format(self.weights.shape)
    linfo += ' activation = {}'.format(self.activation)
    return linfo
		
  def propagate( self, intensor, 
                 return_ext=False, inplace=False ):
    ### calculate the layer output with current weights for a single input tensor
    # input arguments:
    # - return_ext: specify the returned objects.
    #               in the default case, only the output tensor is returned, i.e. f(W.x);
    #               if return_ext is True, the derivative f'(W.x) is returned as well
    # - inplace: if True, the input tensor is modified in-place.
    if not (intensor.is_vector() and intensor.shape[0]==self.input_dim):
      msg = 'Tensor with shape {} is invalid input'.format(intensor.shape)
      msg += ' for DenseLayer with input dimension {}'.format(self.input_dim)
      raise Exception(msg)
    if not inplace:
      if self.biasterm: intensor = intensor.add_bias()
      y = self.weights*intensor
      if not return_ext: return self.activation.f(y)
      else: return (self.activation.f(y), self.activation.df(y))
    else:
      if self.biasterm: intensor.add_bias()
      intensor.leftmultiply_by(self.weights)
      if not return_ext: return self.activation.f(intensor) # todo: add inplace arg
      else: return (self.activation.f(y), self.activation.df(y)) # todo: add inplace arg
	
  def backpropagate( self, intensor, inplace=False ):
    ### calculate the backpropagation tensor for this layer and given input tensor
    # returns a tuple:
    # - the first element is needed when the gradient is calculated w.r.t this layer
    # - the second element is needed for backpropagation further down 
    df = self.propagate(intensor, return_ext=True, inplace=inplace)[1]
    df.diag(inplace=True)
    backweights = df*self.weights
    if self.biasterm: backweights.remove_bias(inplace=True)
    return (df,backweights)
	
  def set_weights( self, weights ):
    ### set the weights for this layer
    # weights can be a Tensor or a numpy array
    if not isinstance(weights,Tensor):
      try: weights = Tensor(weights)
      except:
        msg = 'Object used to set layer weights is not a Tensor'
        msg += ' and could not be casted to one.'
        raise Exception(msg)
    if( self.biasterm and weights.shape!=(self.output_dim,self.input_dim+1) ):
      raise Exception('Weight tensor has wrong shape.')
    if( not self.biasterm and weights.shape!=(self.output_dim,self.input_dim) ):
      raise Exception('Weight tensor has wrong shape.')
    self.weights = weights
		
  def plot_weights(self, fig=None, ax=None, title='' ):
    ### make a plot of the current weight matrix
    # process arguments
    if ax is None: fig,ax = plt.subplots()
    if len(title)==0: title = 'Layer weights'
    # make the plot
    cax = ax.matshow(self.weights.array)
    # print the values if the dimensions are small enough
    if( self.input_dim <= 10 and self.output_dim <= 10):
      for (i, j), z in np.ndenumerate(self.weights.array):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center')
    ax.set_xlabel('input dimension')
    ax.set_ylabel('output dimension')
    plt.title(title)
    plt.colorbar(cax)
    plt.show()
    return (fig,ax)
