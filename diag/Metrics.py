# -*- coding: utf-8 -*-

#####################################################
# metrics and other functionality to test a network #
#####################################################

import numpy as np
import matplotlib.pyplot as plt


class ROC:
	### a class representing a receiver operating characteristic (ROC) curve
	
	def __init__(self,labels,scores):
		nsig = np.sum(labels)
		nback = np.sum(1-labels)
		
		# set score threshold range
		scoremin = np.amin(scores)-1e-7
		# if minimum score is below zero, shift everything up (needed for geomspace)
		if scoremin < 0.: 
		 scores = scores - scoremin + 1.
		 scoremin = 1.
		scoremax = np.amax(scores)+1e-7
    
		scorerange = np.geomspace(scoremin,scoremax,num=100)
		sig_eff = np.zeros(len(scorerange))
		bkg_eff = np.zeros(len(scorerange))
    
		# loop over thresholds
		for i,scorethreshold in enumerate(scorerange):
			sig_eff[i] = np.sum(np.where((labels==1) & (scores>scorethreshold),1,0))/nsig
			bkg_eff[i] = np.sum(np.where((labels==0) & (scores>scorethreshold),1,0))/nback
			
		self.sig_eff = sig_eff[::-1]
		self.bkg_eff = bkg_eff[::-1]
		
	def get_auc(self):
		### calculate and return auc
		auc = np.trapz(self.sig_eff,self.bkg_eff)
		return auc
    
	def plot(self,logx=False):
		### make a plot
		fig,ax = plt.subplots()
		ax.scatter(self.bkg_eff,self.sig_eff)
		ax.set_title('ROC curve')
		ax.set_xlabel('background effiency (background tagged as signal)')
		ax.set_ylabel('signal efficiency (signal tagged as signal)')
		if logx: ax.set_xscale('log')
		# set x axis limits
		ax.set_xlim((np.amin(np.where(self.bkg_eff>0.,self.bkg_eff,1.))/2.,1.))
		# set y axis limits: general case from 0 to 1.
		#ax.set_ylim(0.,1.1)
		# set y axis limits: adaptive limits based on measured signal efficiency array.
		ylowlim = np.amin(np.where((self.sig_eff>0.) & (self.bkg_eff>0.),self.sig_eff,1.))
		ylowlim = 2*ylowlim-1.
		ax.set_ylim((ylowlim,1+(1-ylowlim)/5))
		ax.grid()
		auc = self.get_auc()
		auctext = str(auc)
		if auc>0.99:
			auctext = '1 - '+'{:.3e}'.format(1-auc)
			ax.text(0.7,0.1,'AUC: '+auctext,transform=ax.transAxes)