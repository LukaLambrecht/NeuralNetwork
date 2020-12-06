# -*- coding: utf-8 -*-

######################################################
# testing code for classes in file NetworkHistory.py #
######################################################

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../diag'))
from NetworkHistory import NetworkHistory, NetworkHistoryEntry

history = NetworkHistory()
for i in range(1,11):
	for j in range(1,11):
		metrics = {'accuracy':np.random.rand(),
					 'recall':np.random.rand()}
		entry = NetworkHistoryEntry( epoch=i, batch=j, metrics=metrics )
		history.add_entry(entry)
print(history.metrics)
print(len(history.entries))
history.plot_metrics(do_epoch_axis=True)