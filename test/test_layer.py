# -*- coding: utf-8 -*-

################################
# testing code for Layer class #
################################

# import external modules
import sys
import os
import numpy as np

# import local modules
sys.path.append(os.path.abspath('../src'))
from tensor import Tensor
from layer import DenseLayer

D = DenseLayer(1, 1, 'linear', biasterm=False)
weights = Tensor(np.array([[2]]))
D.set_weights(weights)

intensor = Tensor(np.array([-2]))
outtensor = D.propagate(intensor)
print('intensor: '+str(intensor))
print('outtensor: '+str(outtensor))

(dfdiag,_) = D.backpropagate(intensor)
print('dfdiag: '+str(dfdiag))

D = DenseLayer(1,1,'linear')
weights = Tensor(np.array([[2,2]]))
print(weights.shape)
print(D.weights.shape)
print(D.biasterm)
D.set_weights(weights)

print('intensor: '+str(intensor))
outtensor = D.propagate(intensor)
print('outtensor: '+str(outtensor))

(dfdiag,dfweights) = D.backpropagate(intensor)
print('dfdiag: '+str(dfdiag))
print('dfweights: '+str(dfweights))

D = DenseLayer(10,10,'relu')
print(D)
D.plot_weights()
