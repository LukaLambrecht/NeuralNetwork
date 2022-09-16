# -*- coding: utf-8 -*-

#################################################
# generate toy datasets for testing the network #
#################################################


import numpy as np
import matplotlib.pyplot as plt


def generate_multi_gauss( centers, covs, labels, npoints ):
  ### generate clusters of points according to multivariate normal distributions
  # input arguments:
  # - centers: numpy array of shape (nclusters,ndimensions) with centers
  # - covs: numpy array of shape (nclusters,ndimensions) with covariances
  #         (only diagonal covariances for now)
  # - labels: numpy array of shape (nclusters) with label for each cluster
  # - npoints: integer number of points to generate per cluster
	
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
	
  # generate the clusters
  for label,center,cov in zip(labels,centers,covs):
    cluster = np.random.multivariate_normal(center,np.diag(cov),size=npoints)
    cluster = np.concatenate((np.expand_dims(np.ones(npoints)*label,axis=1),cluster),axis=1)
    clusters.append(cluster)

  # return the result
  clusters = np.concatenate(tuple(clusters),axis=0)	
  return clusters


def plot_clusters( clusters, doshow=False ):
  ### make a plot of data points.
  # input arguments:
  # - clusters: numpy array of shape (npoints,ndims+1)
  #             (where the first column contains the labels)
	
  labels = clusters[:,0]
  points = clusters[:,1:]
	
  if points.shape[1]!=2:
    raise Exception('Only two-dimensional plotting is supported for now.')

  fig,ax = plt.subplots()
  ax.scatter( points[:,0], points[:,1], c=labels )
  if doshow: plt.show()
  return (fig,ax)


if __name__=='__main__':

  # settings
  # define clusters as a list of tuples
  # of format (xcenter, xcov, ycenter, ycov, label)
  clusters = ([
                (1, 0.5, 1, 0.5, 0),
                (3, 0.5, 3, 0.5, 1)
             ])
  npoints = 1000
  outputfile = 'test.npy'

  # parse input
  nclusters = len(clusters)
  centers = np.zeros((nclusters, 2))
  covs = np.zeros((nclusters, 2))
  labels = np.zeros(nclusters)
  for i in range(nclusters):
    centers[i,0] = clusters[i][0]
    centers[i,1] = clusters[i][2]
    covs[i,0] = clusters[i][1]
    covs[i,1] = clusters[i][3]
    labels[i] = clusters[i][4]

  # make and plot clusters
  clusters = generate_multi_gauss( centers, covs, labels, npoints )
  figname = outputfile.replace('.npy','.png')
  (fig,ax) = plot_clusters(clusters)
  fig.savefig(figname, doshow=True)

  # save result
  np.save(outputfile, clusters)
