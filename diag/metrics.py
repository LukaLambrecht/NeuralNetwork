# -*- coding: utf-8 -*-

#####################################################
# metrics and other functionality to test a network #
#####################################################

# import external modules
import numpy as np
import matplotlib.pyplot as plt


class ROC:
    ### a class representing a receiver operating characteristic (ROC) curve
    
    def __init__(self, labels, scores):
        
        # copy attributes
        self.labels = labels
        self.scores = scores

        # calculate number of sig and bkg
        self.nsig = np.sum(self.labels)
        self.nbkg = np.sum(1 - self.labels)
        
        # calculate range
        scoremin = np.amin(self.scores)-1e-7
        scoremax = np.amax(self.scores)+1e-7
        # if minimum score is below zero, define a shift up (needed for geomspace)
        shift = 0
        if scoremin < 0.: shift = 1 - scoremin
        scorerange = np.geomspace(scoremin+shift, scoremax+shift, num=100) - shift
        sig_eff = np.zeros(len(scorerange))
        bkg_eff = np.zeros(len(scorerange))
    
        # loop over thresholds
        for i,scorethreshold in enumerate(scorerange):
            sig_eff[i] = np.sum(np.where((self.labels==1) & (self.scores>scorethreshold),1,0))/self.nsig
            bkg_eff[i] = np.sum(np.where((self.labels==0) & (self.scores>scorethreshold),1,0))/self.nbkg
        self.sig_eff = sig_eff[::-1]
        self.bkg_eff = bkg_eff[::-1]

    def get_sig_scores(self):
        return self.scores[(self.labels==1)]

    def get_bkg_scores(self):
        return self.scores[(self.labels==0)]
        
    def get_auc(self):
        ### calculate and return auc
        auc = np.trapz(self.sig_eff, self.bkg_eff)
        return auc

    def plotscores(self, nbins=30,
            normalize=False, normalizesignal=False,
            siglabel='Signal', sigcolor='g',
            bcklabel='Background', bckcolor='r',
            xaxtitle='Score', xaxtitlesize=12,
            yaxtitle='Frequency', yaxtitlesize=12,
            legendsize=None, legendloc='best',
            ticksize=None):
        ### make a plot of the score distribution
        # define binning between min and max
        minscore = np.min(self.scores)
        maxscore = np.max(self.scores)
        scorebins = np.linspace(minscore,maxscore,num=nbins+1)
        scoreax = (scorebins[1:] + scorebins[:-1])/2
        # split in signal and background
        sigscores = self.get_sig_scores()
        bkgscores = self.get_bkg_scores()
        # make histograms
        sighist = np.histogram(sigscores, bins=scorebins)[0]
        bckhist = np.histogram(bkgscores, bins=scorebins)[0]
        if normalize:
            if np.sum(sighist)!=0: sighist = sighist/np.sum(sighist)
            if np.sum(bckhist)!=0: bckhist = bckhist/np.sum(bckhist)
        if normalizesignal:
            if np.amax(sighist)!=0: sighist *= np.amax(bckhist)/np.amax(sighist)
        # make basic figure
        fig,ax = plt.subplots()
        ax.step(scoreax, bckhist, color=bckcolor, label=bcklabel, where='mid')
        ax.step(scoreax, sighist, color=sigcolor, label=siglabel, where='mid')
        # figure aesthetics
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(ticksize)
        ax.yaxis.get_offset_text().set_fontsize(ticksize)
        if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        ax.legend( loc=legendloc, fontsize=legendsize )
        return (fig,ax)
    
    def plot(self, logx=False,
            xaxtitle='Bkg. efficiency', xaxtitlesize=12,
            yaxtitle='Sig. efficiency', yaxtitlesize=12):
        ### make a plot
        fig,ax = plt.subplots()
        ax.scatter(self.bkg_eff, self.sig_eff)
        if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        if logx: ax.set_xscale('log')
        # set x axis limits
        xlowlim = np.amin(np.where(self.bkg_eff>0., self.bkg_eff, 1.))/2.
        xhighlim = 1.
        ax.set_xlim((xlowlim, xhighlim))
        # set y axis limits
        ylowlim = np.amin(np.where((self.sig_eff>0.) & (self.bkg_eff>0.), self.sig_eff, 1.))/2.
        yhighlim = 1.
        ax.set_ylim((ylowlim, yhighlim))
        ax.grid()
        auc = self.get_auc()
        auctext = '{:.3f}'.format(auc)
        if auc > 0.99:
            auctext = '1 - '+'{:.3e}'.format(1-auc)
        if auc > 1:
            # possible because of round-off errors in numerical integration
            auctext = '1'
        ax.text(0.7, 0.1, 'AUC: '+auctext, transform=ax.transAxes)
