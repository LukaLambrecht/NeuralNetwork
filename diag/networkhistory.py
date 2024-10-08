# -*- coding: utf-8 -*-

#################################################################
# network history object, storing relevant info during training #
#################################################################

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

class NetworkHistoryEntry():
    ### class containing diagnostics info of a network at a given moment in time
    
    def __init__(self, epoch=0, batch=0, metrics={}):
        self.epoch = epoch
        self.batch = batch
        self.metrics = metrics
        
    def get_metric(self, metric, suppress_warning=False):
        ### return the numerical value of a given metric name from this entry
        if metric in self.metrics.keys(): return self.metrics[metric]
        if not suppress_warning:
            msg = 'WARNING: network history entry does not contain requested metric {};'.format(metric)
            msg += ' returning 0...'
            print(msg)
        return 0.

class NetworkHistory():
    ### collection of NetworkHistoryEntries representing the evolution of a network

    def __init__(self):
        self.entries = []
        self.metrics = []
        
    def add_entry(self, entry):
        ### add a NetworkHistoryEntry to the collection
        if not isinstance(entry, NetworkHistoryEntry):
            print('### WARNING ###: history entry cannot be added to network history')
            print('                 skipping this one...')
            return
        for metric in entry.metrics.keys(): 
            if not metric in self.metrics: self.metrics.append(metric)
        self.entries.append(entry)
        
    def add_entry_info(self, epoch=0, batch=0, metrics={}):
        ### add NetworkHistoryEntry without explicitly requiring that class in caller
        self.add_entry( NetworkHistoryEntry(epoch=epoch,batch=batch,metrics=metrics) )

    def get_metric(self, metric, suppress_warning=False):
        ### get the numerical values of a given metric over all entries
        res = np.zeros(len(self.entries))
        for i,entry in enumerate(self.entries):
            res[i] = entry.get_metric(metric, suppress_warning=True)
        return res
        
    def plot_metrics(self, metrics=[], title=None, do_epoch_axis=False):
        ### plot the metrics as a function of timesteps
        ### if metrics is an empty list, all available metrics are plotted
        # set default args
        if len(metrics)==0: metrics = self.metrics
        if title is None: title = 'Metrics during network training'
        # create primary plot
        xax = np.arange(len(self.entries))
        yvals = {}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for metric in metrics:
            yvals[metric] = self.get_metric(metric, suppress_warning=True)
            ax.plot(xax, yvals[metric], label=metric)
        ax.legend()
        ax.set_ylabel('Metric value')
        ax.set_xlabel('Timestep')
        ax.set_title(title)
        if not do_epoch_axis: return (fig,ax)
        # create secondary x-axis with epochs
        epochax = ax.twiny()
        epochxax = [0] # start at zero to align with timestep x-axis
        for i in range(len(self.entries)):
            if self.entries[i].epoch!=self.entries[i-1].epoch:
                epochxax.append(self.entries[i].epoch)
        epochax.plot(epochxax,np.zeros(len(epochxax)),linewidth=0.)
        fig.subplots_adjust(bottom=0.2)
        epochax.xaxis.set_ticks_position('bottom')
        epochax.xaxis.set_label_position('bottom')
        epochax.spines['bottom'].set_position(('axes',-0.15))
        epochax.set_xlabel('epoch')
        return (fig,ax)
