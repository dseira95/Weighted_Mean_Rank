#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Libraries used
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from numpy import array
from lifelines import KaplanMeierFitter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weighted Mean Rank function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Output of WMR function needed as input in C-index function
def MeanRank(survival_time, survival_status, marker):

        keep = survival_time.notnull() & survival_status.notnull() & marker.notnull()
        survival_time = survival_time[keep]
        survival_status = survival_status[keep]
        marker = marker[keep]

        utimes = survival_time[survival_status == 1].unique()
        utimes = sorted(utimes)

        nonparamAUC = np.repeat(None, len(utimes))
        nControls = np.repeat(None, len(utimes))
        TheMarker = marker

        for j in range(0,len(utimes)):
                dead_guy = TheMarker[(survival_time == utimes[j]) & (survival_status == 1)]
                is_started = 0 < utimes[j]
                control_set = TheMarker[(is_started) & (survival_time > utimes[j])]
                set_size = len(control_set)
                nControls[j] = set_size
                ndead = len(dead_guy)
                if ndead == 1:
                        nonparamAUC[j] = sum(np.repeat(dead_guy, set_size) > control_set.values) / set_size
                else:
                        mean_rank = 0
                        for k in range(0,ndead):
                                this_rank = sum(np.repeat(dead_guy[dead_guy.index[k]], set_size) > control_set.values) / set_size
                                mean_rank = mean_rank + this_rank / ndead
                        nonparamAUC[j] = mean_rank

        df = pd.DataFrame()
        df['time'] = pd.DataFrame(utimes)
        df['mean_rank'] = pd.DataFrame(nonparamAUC)
        df['nControls'] = pd.DataFrame(nControls)
        return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# C-index function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Input:
# survival_time = amount of time passed until event occurs
# survival_status = 0 (zero) are censored individuals and 1 (one) means event occurred such as death
# marker = linear predictors from cox model
# cutoffTime = desired end-time point

def dynamicIntegrateAUC(survival_time, survival_status, marker, cutoffTime):

        kmfit = KaplanMeierFitter().fit(survival_time, event_observed=survival_status)
        
        mmm = MeanRank(survival_time=survival_time, survival_status=survival_status, marker=marker)

        meanRanks = mmm.mean_rank[mmm.time <= cutoffTime]
        survTimes = mmm.time[mmm.time <= cutoffTime]

        match = lambda a, b: [ b.index(x) if x in b else None for x in a ] #matching function
        timeMatch = match(survTimes, kmfit.timeline.tolist())
        index = pd.DataFrame(array([i for i in range(0,len(kmfit.survival_function_))]))
        kmfit.survival_function_.insert(len(kmfit.survival_function_.columns), 'new_index', index)
        kmfit.survival_function_.set_index('new_index', inplace=True)
        
        S_t = kmfit.survival_function_.KM_estimate[timeMatch]
        S_t = pd.Series(array(S_t))

        f_t = S_t.drop([len(S_t) - 1]) - pd.Series(array(S_t.drop([0])))
        f_t = pd.Series(np.concatenate(([0],f_t)))
        S_tao = S_t[len(S_t) - 1]
        weights = (2*f_t*S_t) / (1-S_tao**2)

#Output:
# Returns the c-index
        return (sum(meanRanks * weights))
