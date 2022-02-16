#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Libraries used
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from numpy import array, argsort
from lifelines import KaplanMeierFitter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reading the data (PBC data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pbc = pd.read_csv('pbc_data.csv')

#keep the first 312 observations for complete data purposes
pbc = pbc.drop(range(312,418), axis=0)

#remove unnecessary variables (axis=1 for columns; axis=0 for rows)
pbc = pbc.drop(['trt','sex','ascites','hepato','spiders','chol','copper',
                'alk.phos','ast','trig','platelet','stage'], axis=1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Log transformation and censoring transplant patients
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pbc.insert(len(pbc.columns), 'log_bili', np.log(pbc['bili']))
pbc.insert(len(pbc.columns), 'log_protime', np.log(pbc['protime']))

pbc.status = pbc.status.replace(1,0) #change transplant patients to censored(0)
pbc.status = pbc.status.replace(2,1) #change deaths to 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fitting Cox Model with 4 and 5 covariates
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cph_4 = CoxPHFitter().fit(pbc, duration_col='time', event_col='status', 
        formula='age + edema + albumin + log_protime')
cph_5 = CoxPHFitter().fit(pbc, duration_col='time', event_col='status', 
        formula='age + edema + albumin + log_bili + log_protime')
 
 #summary of Cox model
cph_4.print_summary() #4 covariates
cph_5.print_summary() #5 covariates

#parameters from fit model (Betas)
cph_5.params_
cph_4.params_

#linear predictors
betas_xi_5 = pbc.loc[:,['age','edema','albumin','log_bili','log_protime']] * cph_5.params_
preds5 = betas_xi_5.sum(axis=1)
betas_xi_4 = pbc.loc[:,['age','edema','albumin','log_protime']] * cph_4.params_
preds4 = betas_xi_4.sum(axis=1)

pbc.insert(len(pbc.columns), 'predictors_5', preds5)
pbc.insert(len(pbc.columns), 'predictors_4', preds4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scatter Plot of Predictors vs Time (4 and 5 covariates)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, axs = plt.subplots(1, 2, sharex='col')
axs[0].scatter(pbc['time'], pbc['predictors_4'], color=np.where(pbc['status'], '0','r'), marker='.')
axs[0].set_title('4 covariates')
axs[0].set_xlabel('Time (days)')
axs[0].set_ylabel('Predictor Value')
red = mpatches.Patch(color='red', label='Censored')
black = mpatches.Patch(color='black', label='Death')
axs[0].legend(handles=[black, red], loc='best')

axs[1].scatter(pbc['time'], pbc['predictors_5'], color=np.where(pbc['status'], '0','r'), marker='.')
axs[1].set_title('5 covariates')
axs[1].set_xlabel('Time (days)')
axs[1].set_ylabel('Predictor Value')
axs[1].legend(handles=[black, red], loc='best')
fig.suptitle('PBC Data')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Keeping variables used in Weighted Mean Rank fucntion
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
risk = pbc.loc[:, ['time','status','predictors_4','predictors_5']]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weighted Mean Rank function
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

#running WMR function with risk data
wmr_4 = MeanRank(survival_time=risk.time, survival_status=risk.status, marker=risk.predictors_4) #4 covariates
wmr_5 = MeanRank(survival_time=risk.time, survival_status=risk.status, marker=risk.predictors_5) #5 covariates

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scatter Plot of Weighted Mean Rank vs Time
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.scatter(wmr_4['time'], wmr_4['mean_rank'], c='orange', marker='.', label='4 covariates')
plt.scatter(wmr_5['time'], wmr_5['mean_rank'], c='blue', marker='.', label='5 covariates')
plt.title('Time-varying prognostic accuracy')
plt.xlabel('Time (days)')
plt.ylabel('AUC_I/D(t)')
plt.ylim(0.38,1.02)
plt.axhline(0.5, c='grey', ls='--')
plt.legend(loc='best')
plt.show()

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

#running NNE function with WMR data
nne_4 = nne(x=wmr_4.time, y=wmr_4.mean_rank, lambda0=0.2, nControls=wmr_4.nControls) #4 covariates
nne_5 = nne(x=wmr_5.time, y=wmr_5.mean_rank, lambda0=0.2, nControls=wmr_5.nControls) #5 covariates

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scatter Plot of Weighted Mean Rank vs Time with NNE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.scatter(wmr_4['time'], wmr_4['mean_rank'], c='orange', marker='.', label='4 covariates')
plt.scatter(wmr_5['time'], wmr_5['mean_rank'], c='blue', marker='.', label='5 covariates')
plt.title('Time-varying prognostic accuracy')
plt.xlabel('Time (days)')
plt.ylabel('AUC_I/D(t)')
plt.ylim(0.38,1.02)
plt.axhline(0.5, c='grey', ls='--')
plt.legend(loc='best')
plt.plot(nne_4['x'], nne_4['nne'], c='orange')
plt.plot(nne_5['x'], nne_5['nne'], c='blue')
plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# C-index function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

        return (sum(meanRanks * weights))

#running C-index function
dynamicIntegrateAUC(survival_time=risk.time, survival_status=risk.status, marker=risk.predictors_4, cutoffTime=max(risk.time)) #4 covariates
dynamicIntegrateAUC(survival_time=risk.time, survival_status=risk.status, marker=risk.predictors_5, cutoffTime=max(risk.time)) #5 covariates
