# -*- coding: utf-8 -*-

#####################################
# testing code for Activation class #
#####################################

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../src'))
from activation import get_activation
from tensor import Tensor

f = get_activation('linear')
t = Tensor(np.array([[5,-3],[-2,1]]))
print(t)
print(f.f(t))
print(f.df(t))

xax = np.linspace(-3, 3, num=100)
t = Tensor(xax)
all_activations = ['linear', 'relu', 'lrelu', 'sigmoid', 'tanh']
for activation in all_activations:
    f = get_activation(activation)
    fval = f.f(t).array
    dfval = f.df(t).array
    fig,ax = plt.subplots()
    ax.plot(xax, fval, color='b', label='f')
    ax.plot(xax, dfval, color='r', label='df')
    ax.legend()
    ax.set_title(activation)
    fig.show()
    fig.savefig(activation+'.png')
