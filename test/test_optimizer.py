######################################
# Testing code for Optimizer classes #
######################################

# import external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# import local modules
sys.path.append(os.path.abspath('../src'))
from tensor import Tensor
from optimizer import SGD
from optimizer import RMSprop
from optimizer import Adam


def loss(t):
    ### dummy loss function
    # note: keep in sync with dloss function below!
    x,y = t.array
    loss = np.power(x,2) + 3*np.power(y,2)
    return loss

def dloss(t):
    ### dummy loss derivative
    # note: keep in sync with loss function above!
    x,y = t.array
    derivative = Tensor( np.array([2*x, 6*y]) )
    return derivative


# initialize optimizer
#optimizer = SGD(
#  momentum = 0.3,
#  #gradclip = 5
#)
#optimizer = RMSprop(
# learning_rate = 0.1
#)
optimizer = Adam(learning_rate=0.5)

# run the optimizer
init = np.array([3,3])
steps = [init]
t = Tensor(init)
for i in range(100):
    grad = dloss(t)
    t = optimizer.update([t],[grad])[0]
    steps.append(t.array[:,0])

# make a plot
x = [el[0] for el in steps]
y = [el[1] for el in steps]
fig,ax = plt.subplots()
ax.plot(x, y, color='b', linestyle='--', marker='o', markersize=7)
plt.show()
