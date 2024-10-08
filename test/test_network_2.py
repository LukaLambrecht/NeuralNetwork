# -*- coding: utf-8 -*-

############################################
# test the Network class on generated data #
############################################

# import externa modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# network imports
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../diag'))
from layer import DenseLayer
from network import DenseNetwork
from optimizer import SGD,Adam
from metrics import ROC

# data imports
sys.path.append(os.path.abspath('../datagen'))
import generate_data as gen


### STATUS ###
# seems to work well with following settings:
# - labels 0 and 1
# - layers relu or linear in first layer, sigmoid in output layer
# - learning_rate 0.05, momentum 0


# create a dataset
centers = np.array([[0,0],[4,4]])
covs = np.array([[1,1],[1,1]])
categories = np.array([0,1])
clusters = gen.generate_multi_gauss(centers,covs,categories,1000)
gen.plot_clusters( clusters )
np.random.shuffle(clusters)
labels = clusters[:,0]
X_train = clusters[:,1:]
print('shape of training set: '+str(X_train.shape))
print('shape of labels: '+str(labels.shape))

# create a network
N = DenseNetwork()
N.add_layer( DenseLayer(2,2,'linear') )
N.add_layer( DenseLayer(2,1,'sigmoid') )
#N.set_loss_function('mse')
N.set_loss_function('binary_crossentropy')
N.set_optimizer( SGD(learning_rate=0.05, momentum=0.3) )
#N.set_optimizer( Adam(learning_rate=0.01) )
N.set_batch_size(100)
N.set_nepochs(10)
	
# train the network
N.fit(X_train, labels, validation_fraction=0.1)
predictions = N.predict(X_train)

# plot network history
N.history.plot_metrics(do_epoch_axis=True)

# print network weights
N.plot_weights()

# print outputs
nprint = 10
randint = np.random.choice(np.arange(len(labels)),size=nprint)
for i in randint:
    print('label: {} --> prediction: {}'.format(labels[i],predictions[i]))
	
# make a roc curve
roc = ROC(labels, predictions)
roc.plot(logx=True)
plt.show()
