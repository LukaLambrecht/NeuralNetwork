# -*- coding: utf-8 -*-

##########################################
# testing code for file generate_data.py #
##########################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../datagen'))
import generate_data as gen

centers = np.array([[0,0],[5,5]])
covs = np.array([[1,1],[1,1]])
labels = np.array([0,1])
clusters = gen.generate_multi_gauss(centers,covs,labels,1000)
gen.plot_clusters( clusters )