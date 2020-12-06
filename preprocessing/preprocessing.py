# -*- coding: utf-8 -*-

####################################
# data preprocessing functionality #
####################################

import numpy as np

def shuffle_training_set(X_train,labels):
	inds = np.arange(len(labels))
	np.random.shuffle(inds)
	return (X_train[inds],labels[inds])