# -*- coding: utf-8 -*-

###################################
# testing code for file Tensor.py #
###################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../src'))
from Tensor import Tensor


A = Tensor(np.array([[1,2],[3,4]]))
B = Tensor(np.array([[5,6],[7,10]]))
print(A)
print(B)
print(A[0,1])
print(A[1,0])
print(A+B)
print(A*B)
print(A-B)

A = Tensor(np.array([1,2,3]))
print(A)
print(A.shape)
print(A.transpose())
print(A*A.transpose())
print(A.transpose()*A)
print(A.diag())