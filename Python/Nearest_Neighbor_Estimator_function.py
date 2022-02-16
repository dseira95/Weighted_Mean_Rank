#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Libraries used
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from numpy import array

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nearest Neighbor Estimator (NNE) function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_pair_min(y):
        y = sorted(y)
        n = len(y)
        weight = (n - array([i for i in range(1,n+1)]) + 1) + (n - array([i for i in range(1,n+1)]))
        weight = weight / sum(weight)
        mu = sum(y * weight)
        return mu

def nne(x, y, lambda0, nControls=None):
        good_egg = x.notnull() & y.notnull()
        x = x[good_egg]
        y = y[good_egg]

        ooo = np.argsort(x)
        x = x[ooo]
        y = y[ooo]

        n = len(x)

        half_width = n * lambda0 / 2

        nne_estimate = np.repeat(None, n)
        var_estimate = np.repeat(None, n)

        for j in range(0,n):
                iLower = j - half_width
                iUpper = j + half_width
                keep = (array([i for i in range(1,n+1)]) >= iLower) & (array([i for i in range(1,n+1)]) <= iUpper)
                nne_estimate[j] = np.mean(y[keep])
                var_estimate[j] = np.var(y[keep]) / sum(keep)
                if nControls is None:
                        n0 = max(nControls[keep])
                        m = sum(keep)
                        pMin = make_pair_min(y[keep])
                        p = nne_estimate[j]
                        add_term = (pMin - p**2)
                        add_term = add_term - (1/m)*(np.mean(y[keep])) - np.mean(y[keep]**2)
                        var_estimate[j] = var_estimate[j] + add_term/n0

        df = pd.DataFrame()
        df['nne'] = pd.DataFrame(nne_estimate)
        df['var'] = pd.DataFrame(var_estimate)
        df['x'] = pd.DataFrame(x)
        df['lamdba'] = pd.DataFrame(np.repeat(lambda0, n))
        return df