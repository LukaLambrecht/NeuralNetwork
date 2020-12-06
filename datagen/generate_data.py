# -*- coding: utf-8 -*-

#################################################
# generate toy datasets for testing the network #
#################################################

import numpy as np
import matplotlib.pyplot as plt

def generate_multi_gauss(centers,covs,labels,npoints):
	### generate clusters of points according to multivariate normal distributions
	# centers and covs are numpy arrays of shape (nclusters,ndimensions) (only diagonal covariances for now)
	# labels is a numpy array of shape (nclusters)
	# npoints is an integer (number of points to generate per cluster)
	
	# check arguments
	if not centers.shape==covs.shape:
		raise Exception('Centers and covs must have the same shape.')
	if not len(centers.shape)==2:
		raise Exception('Centers and covs must be two-dimensional.')
	if not ( len(labels.shape)==1 and len(labels)==len(centers) ):
		raise Exception('Shape of labels is incompatible with centers and covs.')
		
	# initializations
	nclusters,ndims = centers.shape
	clusters = []
	
	for label,center,cov in zip(labels,centers,covs):
		cluster = np.random.multivariate_normal(center,np.diag(cov),size=npoints)
		cluster = np.concatenate((np.expand_dims(np.ones(npoints)*label,axis=1),cluster),axis=1)
		clusters.append(cluster)
	
	clusters = np.concatenate(tuple(clusters),axis=0)
	
	return clusters

def plot_clusters(clusters):
	### make a plot of data points
	# clusters is a numpy array of shape (npoints,ndims+1),
	# where the first column contains the labels
	
	labels = clusters[:,0]
	points = clusters[:,1:]
	
	if points.shape[1]!=2:
		raise Exception('Only two-dimensional plotting is supported for now.')
		
	plt.scatter(points[:,0],points[:,1],c=labels)
	plt.show()