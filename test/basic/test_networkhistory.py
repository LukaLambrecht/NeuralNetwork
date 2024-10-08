# -*- coding: utf-8 -*-

#########################################
# testing code for class NetworkHistory #
#########################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../../diag'))
from networkhistory import NetworkHistoryEntry
from networkhistory import NetworkHistory

history = NetworkHistory()
for i in range(1,11):
    for j in range(1,11):
        metrics = {'accuracy': np.random.rand(),
                   'recall': np.random.rand()}
        entry = NetworkHistoryEntry( epoch=i, batch=j, metrics=metrics )
        history.add_entry(entry)
print(history.metrics)
print(len(history.entries))
history.plot_metrics(do_epoch_axis=True)
plt.show()
