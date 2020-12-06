# -*- coding: utf-8 -*-

#######################################
# testing code for file Activation.py #
#######################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../src'))
import Activation
from Tensor import Tensor

f = Activation.getActivation('linear')
t = Tensor(np.array([[5,-3],[-2,1]]))
print(t)
print(f.f(t))
print(f.df(t))