# -*- coding: utf-8 -*-

#########################################
# testing code for file LossFunction.py #
#########################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../src'))
from LossFunction import getLossFunction

L = getLossFunction('mse')

labels = np.array([1])
predictions = np.array([2])

print(L.f(labels,predictions))
print(L.df(labels,predictions))