#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Libraries used
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function for Weighted Mean Rank Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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