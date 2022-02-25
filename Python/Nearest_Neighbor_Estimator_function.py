#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Libraries used
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from numpy import array

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nearest Neighbor Estimator (NNE) function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Output of WMR function needed as input 
#Input:
# x = time vector output from weighted mean rank function
# y = weighted mean rank output from WMR function
# lambda0 = half-bandwidth

def nne(x, y, lambda0):

#keeping complete data only
        good_egg = x.notnull() & y.notnull()
        x = x[good_egg]
        y = y[good_egg]

#ordering the data
        ooo = np.argsort(x)
        x = x[ooo]
        y = y[ooo]

        n = len(x)

        half_width = n * lambda0 / 2

        nne_estimate = np.repeat(None, n)
        var_estimate = np.repeat(None, n)

#computing nearest neighbor estimator
        for j in range(0,n):
                iLower = j - half_width
                iUpper = j + half_width
                keep = (array([i for i in range(1,n+1)]) >= iLower) & (array([i for i in range(1,n+1)]) <= iUpper)
                nne_estimate[j] = np.mean(y[keep])
                var_estimate[j] = np.var(y[keep]) / sum(keep)

#Saving the output in a data frame
# nne = nearest neighbor estimator
# var = variance estimate
# x = ordered time event points
# lambda = chosen lambda from input
        df = pd.DataFrame()
        df['nne'] = pd.DataFrame(nne_estimate)
        df['var'] = pd.DataFrame(var_estimate)
        df['x'] = pd.DataFrame(x)
        df['lamdba'] = pd.DataFrame(np.repeat(lambda0, n))
        return df
