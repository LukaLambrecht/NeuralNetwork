# -*- coding: utf-8 -*-

#######################################
# testing code for LossFunction class #
#######################################

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../../src'))
from lossfunction import get_lossfunction

L = get_lossfunction('mse')
labels = np.array([1])
predictions = np.array([2])
print(L.f(labels,predictions))
print(L.df(labels,predictions))

all_lossfunctions = ['mse', 'binary_crossentropy']
xax = np.linspace(-3, 3, num=100)
labels = np.zeros(len(xax))
for lf in all_lossfunctions:
    L = get_lossfunction(lf)
    lval = L.f(labels, xax, domean=False)
    dlval = L.df(labels, xax, domean=False)
    fig,ax = plt.subplots()
    ax.plot(xax, lval, color='b', label='f')
    ax.plot(xax, dlval, color='r', label='df')
    ax.legend()
    ax.set_title(lf)
plt.show()
