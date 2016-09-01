"""
Created on Thu Oct 16 10:48:19 2014
Modified August 2016 to include horse-race model (HRM)

@author: Hayley
Code for activation threshold and horse-race models. Using ARI task experimental data.

To run ATM on Go and GS trials with a single facilitation curve: python ARI_task.py ATM_first
To run ATM on GS trials with an additional facilitation curve using manual input
for optimized parameters from single facilitation curve: python ARI_task.py ATM_second
To run HRM on Go and GS trials: python ARI_task.py HRM

"""

import numpy as np
from scipy import stats
import scipy.optimize as opt
import sys

#%%
def get_trials(params, n_rep=10000):  
    '''
    Generates n_rep Guassian facilitation curves 
    
    Parameters
    -------------
    params : sequence (8,) of float
        mean and sd for a_facGo, b_facGo, c_facGo, inhib
    n_rep : flat
        number of simulated trials
        
    Returns
    --------
    fac_i : array (t, n_rep)
        facilitation curves for all simulated trials
    inhib_tonic : array (t, n_rep)
        horizontal lines denoting tonic inhibition for all simulated trials
    t : array (600,)
        sequence of time index
    '''
    a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, inhib_mean, inhib_sd = params  
    t = np.linspace(-.4, .2, 600, endpoint=False)  
    a_facGo = np.random.normal(a_facGo_mean, a_facGo_sd, size=n_rep) 
    b_facGo = np.random.normal(b_facGo_mean, b_facGo_sd, size=n_rep)
    c_facGo = np.random.normal(c_facGo_mean, c_facGo_sd, size=n_rep)
    fac_i = np.zeros((n_rep, t.size))  
    inhib_tonic = np.zeros((n_rep, t.size))    
    inhib = np.random.normal(inhib_mean, inhib_sd, size=n_rep)
    inhib_tonic += inhib[:,np.newaxis]
        
    for i in range(n_rep):  
        myparams_fac = a_facGo[i], b_facGo[i], c_facGo[i]  
        fac_i[i] = get_fac(t, myparams_fac)  
    return fac_i, inhib_tonic, t

#%%
def get_trials_HRM(params, n_rep=10000):  
    '''
    Generates n_rep Guassian facilitation curves 
    
    Parameters
    -------------
    params : sequence (7,) of float
        mean and sd for a_facGo, b_facGo, c_facGo, a single decision threshold value
    n_rep : flat
        number of simulated trials
        
    Returns
    --------
    fac_i : array (t, n_rep)
        facilitation curves for all simulated trials
    decision_threshold : array (t, n_rep)
        horizontal lines denoting fixed threshold for all simulated trials
    t : array (600,)
        sequence of time index
    '''
    a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, threshold_value = params  
    t = np.linspace(-.4, .2, 600, endpoint=False)  
    a_facGo = np.random.normal(a_facGo_mean, a_facGo_sd, size=n_rep) 
    b_facGo = np.random.normal(b_facGo_mean, b_facGo_sd, size=n_rep)
    c_facGo = np.random.normal(c_facGo_mean, c_facGo_sd, size=n_rep)
    fac_i = np.zeros((n_rep, t.size))  
    decision_threshold = np.ones((n_rep, t.size))   
    decision_threshold = decision_threshold * threshold_value
            
    for i in range(n_rep):  
        myparams_fac = a_facGo[i], b_facGo[i], c_facGo[i]  
        fac_i[i] = get_fac(t, myparams_fac)  
    return fac_i, decision_threshold, t

#%%
def get_inhib_curves_HRM(params, n_rep=10000):  
    '''
    Generates n_rep Guassian inhibition curves 
    
    Parameters
    -------------
    params : sequence (6,) of float
        mean and sd for a_inhibGS, b_inhibGS, c_inhibGS
    n_rep : flat
        number of simulated trials
        
    Returns
    --------
    inhib_i : array (t, n_rep)
        inhibition curves for all simulated trials
    '''
    a_inhib_mean, a_inhib_sd, b_inhib_mean, b_inhib_sd, c_inhib_mean, c_inhib_sd = params  
    t = np.linspace(-.4, .2, 600, endpoint=False)  
    a_inhibGS = np.random.normal(a_inhib_mean, a_inhib_sd, size=n_rep) 
    b_inhibGS = np.random.normal(b_inhib_mean, b_inhib_sd, size=n_rep)
    c_inhibGS = np.random.normal(c_inhib_mean, c_inhib_sd, size=n_rep)
    inhib_i = np.zeros((n_rep, t.size))  
                
    for i in range(n_rep):  
        myparams_fac = a_inhibGS[i], b_inhibGS[i], c_inhibGS[i]  
        inhib_i[i] = get_fac(t, myparams_fac)  # use same function as facilitation curves as same Gaussian function
    return inhib_i

#%%
def get_fac(t, params):
    '''
    Generates a single Gaussian facilitation curve
    
    Parameters
    -------------
    params : sequence (3,) of float
        values for a_facGo, b_facGo, c_facGo
        
    Returns
    -------
    fac : array (t,)
        facilitation curve values at times `t`
    '''
    a_facGo, b_facGo, c_facGo = params
    fac = a_facGo * np.exp(-(t - b_facGo)**2 /(2 * c_facGo**2))
    return fac

#%%
def get_activation_thresholds(t, inhib_tonic, params_GS, n_rep=10000): 
    '''
    Generates n_rep inhibition curves i.e. activation thresholds
    
    Parameters
    -------------
    params : sequence (4,) of float
        values for k_inhib, tau_inhib, mean and sd for step_t
        
    Returns
    --------
    thresholds : array (t, n_rep)
        inhibition curves for all simulated trials
    '''
    k_inhib, tau_inhib, step_t_mean, step_t_sd = params_GS
    thresholds = np.zeros((n_rep, t.size))
    for i in range(n_rep):
        thresholds[i] = get_inhib_increase(t, inhib_tonic[i], params_GS)
    return thresholds

#%%
def get_inhib_increase(t, inhib_tonic, params_GS):
    '''
    Generates a single inhibition curve
    
    Returns
    ----------
    inhib : array (t,)
        inhibition curve values at times `t`
    '''
    k_inhib, tau_inhib, step_t_mean, step_t_sd = params_GS 
    step_t = np.random.normal(step_t_mean, step_t_sd) 
    inhib = k_inhib * (1 - np.exp(-(t+step_t)/tau_inhib)) + inhib_tonic 
    inhib = np.maximum(inhib, inhib_tonic) 
    return inhib

#%% 
def get_trials_facNew(params_facNew, facBimanual, t, n_rep=10000): 
    '''
    Generates n_rep facilitation curves from additive Gaussian functions
    
    Parameters
    -------------
    params_facNew : sequence (2,) of float
        b_facGo_mean, b_facGo_sd for second facilitation curve 
    facBimanual : array (t, n_rep)
        original facilitation curves 
            
    Returns
    --------
    fac_i_new : array (t, n_rep)
        combined facilitation curves for all simulated trials
    '''
    a_facGo_mean = 2.6
    a_facGo_sd = 0.03 
    b_facGo_mean, b_facGo_sd = params_facNew
    c_facGo_mean = 0.06
    c_facGo_sd = 0.01  
    a_facGo = np.random.normal(a_facGo_mean, a_facGo_sd, size=n_rep) 
    b_facGo = np.random.normal(b_facGo_mean, b_facGo_sd, size=n_rep)
    c_facGo = np.random.normal(c_facGo_mean, c_facGo_sd, size=n_rep)
    fac_i_new = np.zeros((n_rep, t.size))
    
    for i in range(n_rep):
        myparams_fac = a_facGo[i], b_facGo[i], c_facGo[i]
        fac_i_new[i] = get_facNew(t, myparams_fac, facBimanual[i])
    return fac_i_new

#%%
def get_facNew(t, params_new, facBimanual):
    '''
    Generates a single facilitation curve from additive Gaussian functions
    
    Parameters
    ----------
    params_new : sequence (3,) of float
        values for a_facGo, b_facGo, c_facGo
            
    Returns
    -------
    fac2 : array (t,)
        combined facilitation curve values at times `t`
    '''    
    fac1 = facBimanual
    a_facGo, b_facGo, c_facGo = params_new
    fac2 = np.add(fac1, (a_facGo * np.exp(-(t - b_facGo)**2 /(2 * c_facGo**2))))
    return fac2

#%%   
def get_fac_tms_vals(t, fac_i, pts=(-.15, -.125, -.1)):
    '''
    Gets values at pre-defined time points on all simulated facilitation curves
    
    Parameters 
    -------------
    pts : sequence of floats
        time points for comparison to MEP amplitude values from experimental data
        
    Returns
    -------------
    valsXX : arrays, each of (n_rep,) 
        value on all simulated facilitation curves at time point (in ms) requested
    '''
    idx150 = np.flatnonzero(np.isclose(t, pts[0]))
    vals150 = fac_i[:,idx150]
    idx125 = np.flatnonzero(np.isclose(t, pts[1]))
    vals125 = fac_i[:,idx125]
    idx100 = np.flatnonzero(np.isclose(t, pts[2]))
    vals100 = fac_i[:,idx100]
    return (vals150, vals125, vals100)  

#%%    
def get_GS_tms_vals(t, fac_i, activation_thresholds, inhib_tonic, pts=(-0.075, -0.05, -0.025)):
    '''
    Gets values at pre-defined time points as difference between facilitation
    curves and rise in inhibition curve above baseline
    
    Returns
    -------------
    predsXX : arrays, each of (n_rep,)
        values for difference at time point (in ms) requested
    '''
    index75 = np.flatnonzero(np.isclose(t, pts[0]))
    fac_values75 = fac_i[:, index75]
    inhib_step_values75 = activation_thresholds[:, index75]
    diff_inhib75 = inhib_step_values75 - inhib_tonic 
    pred75 = fac_values75 - diff_inhib75 
    index50 = np.flatnonzero(np.isclose(t, pts[1]))
    fac_values50 = fac_i[:, index50]
    inhib_step_values50 = activation_thresholds[:, index50]
    diff_inhib50 = inhib_step_values50 - inhib_tonic
    pred50 = fac_values50 - diff_inhib50
    index25 = np.flatnonzero(np.isclose(t, pts[2]))
    fac_values25 = fac_i[:, index25]
    inhib_step_values25 = activation_thresholds[:, index25]
    diff_inhib25 = inhib_step_values25 - inhib_tonic
    pred25 = fac_values25 - diff_inhib25    
    return pred75, pred50, pred25

#%%    
def get_GS_tms_vals_HRM(t, fac_i, inhib_i, pts=(-0.075, -0.05, -0.025)):
    '''
    Gets values at pre-defined time points as difference between facilitation
    curves and rise in inhibition curve above baseline
    
    Returns
    -------------
    predsXX : arrays, each of (n_rep,)
        values for difference at time point (in ms) requested
    '''
    index75 = np.flatnonzero(np.isclose(t, pts[0]))
    fac_values75 = fac_i[:, index75]
    inhib_values75 = inhib_i[:, index75]
    pred75 = fac_values75 - inhib_values75 
    index50 = np.flatnonzero(np.isclose(t, pts[1]))
    fac_values50 = fac_i[:, index50]
    inhib_values50 = inhib_i[:, index50]
    pred50 = fac_values50 - inhib_values50
    index25 = np.flatnonzero(np.isclose(t, pts[2]))
    fac_values25 = fac_i[:, index25]
    inhib_values25 = inhib_i[:, index25]
    pred25 = fac_values25 - inhib_values25    
    return pred75, pred50, pred25

#%%
def get_emg_onsets_offsets(t, fac_i, inhib): 
    '''
    Gets times when inhibition and facilitaiton curves intersect and slope of
    facilitation curve at first point of intersection. If facilitation curve
    doesn't cross inhibition curve twice (onset & offset), generates large error.
    
    Parameters
    -------------
    inhib : array (t,n_rep)
        inhibition curves for all simulated trials
    
    Returns
    -------------
    emg_onsets : array (n_rep,)
        time point when facilitation curves first rise above inhibition/threshold
    gradient : array (n_rep,)
        slope of facilitation curves at first intersection
    emg_offsets : array (n_rep,)
        time point when facilitation curves drop below inhibition/threshold
    '''
    ntrials = fac_i.shape[0]
    gradient = np.zeros(ntrials) + np.nan
    getinhib = fac_i < inhib 
    switches = getinhib.astype(int)
    switches_diff = np.diff(switches) 
    index_trials_onsets = np.nonzero(switches_diff == -1)  
    index_trials_offset = np.nonzero(switches_diff == 1)         
    emg_onsets = t[index_trials_onsets[1]]
    emg_offsets= t[index_trials_offset[1]]
    for i in range(ntrials):
        if np.all(switches[i] == 1):
            emg_onsets = np.append(emg_onsets, (1000 * (inhib[i,1] - fac_i[i].max()) + t[np.argmax(fac_i[i])]))
            emg_offsets = np.append(emg_offsets, (1000 * (inhib[i,1] - fac_i[i].max()) + t[np.argmax(fac_i[i])]))
        elif switches[i, -1] == 0:
            emg_offsets = np.append(emg_offsets, (1000 * (fac_i[i, -1] - inhib[i, -1]) + t[-1]))
   
    for trial, time_pt in zip(index_trials_onsets[0], index_trials_onsets[1]):
        rise = fac_i[trial, time_pt + 1] - fac_i[trial, time_pt - 1]
        run  = t[time_pt + 1] - t[time_pt - 1]
        gradient[trial] = rise / run
        if run == 0:
            print "Error - run equals zero"
    return emg_onsets, gradient, emg_offsets 

#%% 
def get_emg_onsets_facNew(t, fac_i, inhib): 
    '''
    Gets time when inhibition and facilitaiton curves first intersect and 
    slope of facilitation curve at point of intersection. If facilitation curve
    doesn't cross inhibition curve, generates large error.
    '''

    ntrials = fac_i.shape[0]
    gradient = np.zeros(ntrials) + np.nan
    getinhib = fac_i < inhib 
    switches = getinhib.astype(int)
    switches_diff = np.diff(switches) 
    index_trials_onsets = np.nonzero(switches_diff == -1)  
    emg_onsets = t[index_trials_onsets[1]]
    for i in range(ntrials):
        if np.all(switches[i] == 1):
            emg_onsets = np.append(emg_onsets, (1000 * (inhib[i,1] - fac_i[i].max()) + t[np.argmax(fac_i[i])]))
    
    for trial, time_pt in zip(index_trials_onsets[0], index_trials_onsets[1]):
        rise = fac_i[trial, time_pt + 1] - fac_i[trial, time_pt - 1]
        run  = t[time_pt + 1] - t[time_pt - 1]
        gradient[trial] = rise / run        
    return emg_onsets, gradient 

#%%
def get_chisquare(obs_data, obs_model, nbins=3):
    '''
    Calculates histograms for experimental and predicted data. 
    Compares frequencies in each bin. Calculates one-way Chi-square test.
    
    Parameters
    --------------
    obs_data : array, 1-D, length of number of data points
        experimental data values
    obs_model : array (n_rep,)
        predicted values from simulated trials 
    
    Returns
    ---------------
    Chi-square :  float
        Chi-square statistic for how well predicted data matches experimental data    
    '''
    percentile_bins = np.linspace(0, 100, nbins + 1)    
    bin_edges = np.percentile(obs_data, list(percentile_bins))
    hist_data, bin_edges  = np.histogram(obs_data,  bins=bin_edges)
    hist_data = hist_data / float(obs_data.size)  
    hist_model, bin_edges = np.histogram(obs_model, bins=bin_edges)
    hist_model = hist_model / float(obs_model.size)
    return stats.chisquare(hist_data, hist_model)

#%%
def error_function_Go(params, data150, data125, data100, data_onsets, data_offsets):  
    '''
    Compares experimental Go trial MEP and EMG data to values predicted from 
    simulated Go trials. Calculates summed Chi-square.  
    
    Parameters
    --------------
    params : sequence (8,) of float
        current mean and sd values for a_facGo, b_facGo, c_facGo, inhib
    dataXX : arrays, 1-D, each length of number of data points
        experimental Go trial MEP amplitudes and EMG onsets & offsets 
     
    Returns
    ---------------
    X2_summed_Go : float
        statistic for how well predicted data matches experimental Go trial data    
    '''
    print "Trying with values: " + str(params) 
    fac_i, inhib_tonic, t = get_trials(params)
    pred150, pred125, pred100 = get_fac_tms_vals(t, fac_i)    
    pred_onsets, pred_rates, pred_offsets = get_emg_onsets_offsets(t, fac_i, inhib_tonic) 
    X2_onsets = get_chisquare(data_onsets, pred_onsets, nbins=2)[0]
    print "X2_onsets: ", X2_onsets
    X2_offsets = get_chisquare(data_offsets, pred_offsets, nbins=2)[0]
    print "X2_offsets: ", X2_offsets    
    X2_150 = get_chisquare(data150, pred150, nbins=2)[0]
    print "X2_150: ", X2_150
    X2_125 = get_chisquare(data125, pred125, nbins=2)[0]
    print "X2_125: ", X2_125
    X2_100 = get_chisquare(data100, pred100, nbins=2)[0]
    print "X2_100: ", X2_100
    X2_summed_Go = X2_150 + X2_125 + X2_100 + X2_onsets + X2_offsets
    print "X2 summed: ", X2_summed_Go
    return X2_summed_Go
     
#%%
def error_function_GS(params_GS, params_Go, data75, data50, data25):
    '''
    Compares experimental GS trial MEP data to values predicted from 
    simulated GS trials with single facilitaiton curve and raised threshold. 
    Calculates summed Chi-square.
    
    Parameters
    --------------
    params_GS : sequence (4,) of float
        current values for k_inhib and tau_inhib, mean & sd for step_t
    params_Go : tuple (3,) of arrays, each of shape (t, nreps)
        fac_i, inhib_tonic, t generated from optimized Go parameters
    dataXX : arrays, 1-D, each length of number of data points
        experimental GS trial MEP amplitudes 
    
    Returns
    -------
    X2_summed_GS : float
        statistic for how well predicted MEP data matches experimental GS trial data
    '''
    print "Trying with values: " + str(params_GS)
    fac_i, inhib_tonic, t = params_Go
    activation_thresholds = get_activation_thresholds(t, inhib_tonic, params_GS) 
    pred75, pred50, pred25 = get_GS_tms_vals(t, fac_i, activation_thresholds, inhib_tonic)
    X2_75 = get_chisquare(data75, pred75, nbins=2)[0]
    print "X2_75: ", X2_75
    X2_50 = get_chisquare(data50, pred50, nbins=2)[0]
    print "X2_50: ", X2_50
    X2_25 = get_chisquare(data25, pred25, nbins=2)[0]
    print "X2_25: ", X2_25
    X2_summed_GS = X2_75 + X2_50 + X2_25
    print "X2_summed: ", X2_summed_GS
    return X2_summed_GS

#%%
def error_function_GS_facNew(params_facNew, activation_thresholds, components_Go, data_onsets, data_rates):
    '''
    Compares experimental GS trial EMG onset time and rate of onset data to 
    values predicted from simulated GS trials with additive facilitaiton curves. 
    Calculates summed Chi-square.
    
    Parameters
    --------------
    components_Go : tuple (3,) of arrays, each (t,n_rep)
        fac_i, inhib_tonic, t generated from optimized Go parameters
    dataXX : arrays, 1D, each length of data points
        experimental GS trial EMG onset times and rates 
    
    Returns
    ---------------
    X2_summed_facNew : float
        statistic for how well predicted EMG data matches experimental GS trial 
        data with combined facilitatory input    
    '''
    print "Trying with value: " + str(params_facNew)
    fac_i, inhib_tonic, t = components_Go
    fac_i_new = get_trials_facNew(params_facNew, components_Go[0], t)
    pred_onsets, pred_rates = get_emg_onsets_facNew(t, fac_i_new, activation_thresholds)
    X2_onsets = get_chisquare(data_onsets, pred_onsets, nbins=2)[0]
    print "X2_onsets: ", X2_onsets
    X2_rates = get_chisquare(data_rates, pred_rates, nbins=2)[0]
    print "X2_rates: ", X2_rates
    X2_summed_facNew = X2_onsets + X2_rates
    print "X2_summed: ", X2_summed_facNew
    return X2_summed_facNew
 
#%%
def error_function_Go_HRM(params, data150, data125, data100, data_onsets, data_offsets):  
    '''
    Compares experimental Go trial MEP and EMG data to values predicted from 
    simulated Go trials. Calculates summed Chi-square.  
    
    Parameters
    --------------
    params : sequence (7,) of float
        current mean and sd values for a_facGo, b_facGo, c_facGo, and single value for tonic decision threshold 
    dataXX : arrays, 1-D, each length of number of data points
        experimental Go trial MEP amplitudes and EMG onsets & offsets 
     
    Returns
    ---------------
    X2_summed_Go : float
        statistic for how well predicted data matches experimental Go trial data    
    '''
    print "Trying with values: " + str(params) 
    fac_i, decision_threshold, t = get_trials_HRM(params)
    pred150, pred125, pred100 = get_fac_tms_vals(t, fac_i)    
    pred_onsets, pred_rates, pred_offsets = get_emg_onsets_offsets(t, fac_i, decision_threshold) 
    X2_onsets = get_chisquare(data_onsets, pred_onsets, nbins=2)[0]
    print "X2_onsets: ", X2_onsets
    X2_offsets = get_chisquare(data_offsets, pred_offsets, nbins=2)[0]
    print "X2_offsets: ", X2_offsets    
    X2_150 = get_chisquare(data150, pred150, nbins=2)[0]
    print "X2_150: ", X2_150
    X2_125 = get_chisquare(data125, pred125, nbins=2)[0]
    print "X2_125: ", X2_125
    X2_100 = get_chisquare(data100, pred100, nbins=2)[0]
    print "X2_100: ", X2_100
    X2_summed_Go = X2_150 + X2_125 + X2_100 + X2_onsets + X2_offsets
    print "X2 summed: ", X2_summed_Go
    return X2_summed_Go
        
#%%
def error_function_GS_HRM(params_GS, params_Go, data75, data50, data25, data_onsets):
    '''
    Compares experimental GS trial MEP data to values predicted from 
    simulated GS trials. 
    Calculates summed Chi-square.
    
params_GS = []     
    
    
    Parameters
    --------------
    params_GS : sequence (6,) of float
        current mean & sd for a_inhib, b_inhib, c_inhib
    params_Go : tuple (3,) of arrays, each of shape (t, nreps)
        fac_i, decision_threshold, t generated from optimized Go parameters
    dataXX : arrays, 1-D, each length of number of data points
        experimental GS trial MEP amplitudes 
    
    Returns
    -------
    X2_summed_GS : float
        statistic for how well predicted MEP data matches experimental GS trial data
    '''
    print "Trying with values: " + str(params_GS)
    fac_i, decision_threshold, t = params_Go
    inhib_i = get_inhib_curves_HRM(params_GS)
    pred_onsets_Go, pred_rates_Go, pred_offsets_Go = get_emg_onsets_offsets(t, fac_i, decision_threshold)
    data_rates_GS = np.multiply(pred_rates_Go, 1.2)        
    pred75, pred50, pred25 = get_GS_tms_vals_HRM(t, fac_i, inhib_i)
    pred_onsets, pred_rates = get_emg_onsets_facNew(t, fac_i, decision_threshold)
    data_rates_GS = data_rates_GS[np.isfinite(data_rates_GS)] # removes any nan and infinite from array i.e. fac curve has reached but not passed threshold
    pred_rates = pred_rates[np.isfinite(pred_rates)]
    X2_onsets = get_chisquare(data_onsets, pred_onsets, nbins=2)[0]
    print "X2_onsets: ", X2_onsets
    X2_rates = get_chisquare(data_rates_GS, pred_rates, nbins=2)[0]
    print "X2_rates: ", X2_rates
    X2_75 = get_chisquare(data75, pred75, nbins=2)[0]
    print "X2_75: ", X2_75
    X2_50 = get_chisquare(data50, pred50, nbins=2)[0]
    print "X2_50: ", X2_50
    X2_25 = get_chisquare(data25, pred25, nbins=2)[0]
    print "X2_25: ", X2_25
    X2_summed_GS = X2_75 + X2_50 + X2_25 + X2_onsets + X2_rates
    print "X2_summed: ", X2_summed_GS
    return X2_summed_GS

#%%    
def load_exp_data(fname):
    '''
    Gets data values from .csv file and removes NaNs
    
    Parameters
    --------------
    fname : string
        file path for experimental data 
        
    Returns
    ---------------
    no_nan_data : array, 1-D, length of data points
        experimental data values with NaNs removed
    '''
    file_contents = np.genfromtxt(fname, dtype=float, delimiter=",")  
    data  = file_contents.flatten()
    no_nan_data = data[~np.isnan(data)] 
    return no_nan_data
    
#%%
# Loads experimental data
data_dir = ''

# Go trials
fname150 = data_dir + 'Go_MEP_amplitudes_150ms.csv'
fname125 = data_dir + 'Go_MEP_amplitudes_125ms.csv'
fname100 = data_dir + 'Go_MEP_amplitudes_100ms.csv'
exp_MEPs_150 = load_exp_data(fname150)
exp_MEPs_125 = load_exp_data(fname125)
exp_MEPs_100 = load_exp_data(fname100)
fnameGoThreeStimOnly = data_dir + 'Go_EMG_onsets.csv'
exp_EMG_onsets_three_stim = load_exp_data(fnameGoThreeStimOnly) / 1000 - .8  # /1000 to convert onset times into seconds, -0.8 to set relative to target line at 0ms
sim_data_Go_EMG_offsets = np.add(exp_EMG_onsets_three_stim, 0.107)  # sets EMG offset times based on empirical average burst duration of 107 ms

# GS trials
fnameGS75 = data_dir + 'GS_MEP_amplitudes_75ms.csv'
fnameGS50 = data_dir + 'GS_MEP_amplitudes_50ms.csv'
fnameGS25 = data_dir + 'GS_MEP_amplitudes_25ms.csv'
exp_GS_MEPs_75 = load_exp_data(fnameGS75)
exp_GS_MEPs_50 = load_exp_data(fnameGS50)
exp_GS_MEPs_25 = load_exp_data(fnameGS25)
fnameGSStimOnsets = data_dir + 'GS_EMG_onsets.csv'
exp_GS_EMG_onsets_three_stim = load_exp_data(fnameGSStimOnsets) / 1000 - .8
    
#%%
def build_Go_GS_ATM():
    '''
    Optimizes parameters for facilitation and inhibtiion curves on Go and 
    GS trials in Activation Threshold Model (with single facilitation curve)
    
    params_Go : 
        a_facGo_mean - average amplitude of Gaussian curve
        a_facGo_sd - standard deviation for amplitude of Gaussian curve
        b_facGo_mean - average time to peak of Gaussian curve
        b_facGo_sd - standard deviation for time to peak of Gaussian curve
        c_facGo_mean - average curvature of Gaussian curve
        c_facGo_sd - standard deviation for curvature of Gaussian curve
        inhib_mean - average value for tonic inhibition
        inhib_sd - standard deviation for tonic inhibition
    params_GS : 
        k_inhib - amplitude of step function
        tau_inhib - time constant of step function
        step_t_mean - 't' at step function onset
        step_t_sd - variation in step function onset
    '''
    # Example initial guess for Go parameters
    a_facGo_mean = 2
    a_facGo_sd = 0.2
    b_facGo_mean = 0.06
    b_facGo_sd = 0.02
    c_facGo_mean = 0.12
    c_facGo_sd = 0.01
    inhib_mean = 1.5
    inhib_sd = 0.3
    # Example initial guess for GS parameters
    k_inhib = 1.2
    tau_inhib = 0.8
    step_t_mean = 0.1
    step_t_sd = 0.02
    
    params_Go = [a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, inhib_mean, inhib_sd]
    optGo = opt.minimize(error_function_Go, params_Go, args=(exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim, sim_data_Go_EMG_offsets), method='Nelder-Mead', tol=0.01)  
    print "ParamsOptimizedGo", optGo 
    params_Go = optGo.x # optimized parameter output from Go optimization function 
    fac_i, inhib_tonic, t = get_trials(params_Go) 
    components_Go = (fac_i, inhib_tonic, t)
    params_GS = [k_inhib, tau_inhib, step_t_mean, step_t_sd]   
    optGS  = opt.minimize(error_function_GS, params_GS, args=(components_Go, exp_GS_MEPs_75, exp_GS_MEPs_50, exp_GS_MEPs_25), method='Nelder-Mead', tol=0.01)    
    print "ParamsOptimizedGS", optGS

#%%
def build_GS_facNew():
    '''
    Generates original facilitation and inhibition curves from previously optimized 
    parameters. Sets EMG rate of onset for GS trials at 120% of Go trial values, 
    as observed experimentally. Optimizes b_facGoNew mean and sd values for 
    second facilitation curve on GS trials.
    
    params_facNew : 
        b_facGS_new_mean - average time to peak of second Gaussian curve
        b_facGS_new_sd - standard deviation for time to peak of second Gaussian curve
    '''
    # Optimized Go parameters
    a_facGo_mean = 2.57619299
    a_facGo_sd = 0.05308114
    b_facGo_mean = 00640317
    b_facGo_sd = 0.00782825
    c_facGo_mean = 0.06399041
    c_facGo_sd = 0.01087128
    inhib_mean = 1.79788944
    inhib_sd = 0.24566962
    # Optimized GS parameters
    k_inhib = 1.88698162
    tau_inhib = 0.05998707
    step_t_mean = 0.13339161
    step_t_sd = 0.01796852
    # Example initial guess for GS second facilitation curve (params_facNew)
    b_facGS_new_mean = 0.2
    b_facGS_new_sd = 0.05
    
    params_Go = [a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, inhib_mean, inhib_sd] 
    fac_i, inhib_tonic, t = get_trials(params_Go)
    components_Go = (fac_i, inhib_tonic, t)
    pred_onsets, pred_rates, pred_offsets = get_emg_onsets_offsets(t, fac_i, inhib_tonic) 
    sim_data_GS_rates = np.multiply(pred_rates, 1.2) 
    params_GS = [k_inhib, tau_inhib, step_t_mean, step_t_sd] 
    activation_thresholds = get_activation_thresholds(t, inhib_tonic, params_GS)
    params_facNew = [b_facGS_new_mean, b_facGS_new_sd]  
    optFacNew = opt.minimize(error_function_GS_facNew, params_facNew, args=(activation_thresholds, components_Go, exp_GS_EMG_onsets_three_stim, sim_data_GS_rates), method='Nelder-Mead', tol=0.01)
    print "ParamsOptimizedGSFacNew", optFacNew  

#%%
def build_Go_GS_HRM():
    '''
    Optimizes parameters for facilitation and inhibtiion curves on Go and 
    GS trials in Horse-Race Model (HRM)
    
    params_Go : 
        a_facGo_mean - average amplitude of Gaussian curve
        a_facGo_sd - standard deviation for amplitude of Gaussian curve
        b_facGo_mean - average time to peak of Gaussian curve
        b_facGo_sd - standard deviation for time to peak of Gaussian curve
        c_facGo_mean - average curvature of Gaussian curve
        c_facGo_sd - standard deviation for curvature of Gaussian curve
        threshold - set value for decision threshold
        
    params_GS : 
        a_inhib_mean - average amplitude of Gaussian curve
        a_inhib_sd - standard deviation for amplitude of Gaussian curve
        b_inhib_mean - average time to peak of Gaussian curve
        b_inhib_sd - standard deviation for time to peak of Gaussian curve
        c_inhib_mean - average curvature of Gaussian curve
        c_inhib_sd - standard deviation for curvature of Gaussian curve
    '''
#    # Example initial guess for Go parameters (facilitation curves and decision threshold)
    a_facGo_mean = 2
    a_facGo_sd = 0.2
    b_facGo_mean = 0.06
    b_facGo_sd = 0.02
    c_facGo_mean = 0.12
    c_facGo_sd = 0.01
    threshold = 1.5
    
#    # Example initial guess for inhibition curve parameters on GS trials
    a_inhib_mean = 1.3 
    a_inhib_sd = 0.16
    b_inhib_mean = 0.02
    b_inhib_sd = 0.01
    c_inhib_mean = 0.06
    c_inhib_sd = 0.01
    
    params_Go = [a_facGo_mean, a_facGo_sd, b_facGo_mean, b_facGo_sd, c_facGo_mean, c_facGo_sd, threshold]
    optGo = opt.minimize(error_function_Go_HRM, params_Go, args=(exp_MEPs_150, exp_MEPs_125, exp_MEPs_100, exp_EMG_onsets_three_stim, sim_data_Go_EMG_offsets), method='Nelder-Mead', tol=0.01)  
    print "ParamsOptimizedGo", optGo 
    params_Go = optGo.x # optimized parameter output from Go optimization function 
    fac_i, decision_threshold, t = get_trials_HRM(params_Go) 
    components_Go = (fac_i, decision_threshold, t)
    params_GS = [a_inhib_mean, a_inhib_sd, b_inhib_mean, b_inhib_sd, c_inhib_mean, c_inhib_sd]   
    optGS  = opt.minimize(error_function_GS_HRM, params_GS, args=(components_Go, exp_GS_MEPs_75, exp_GS_MEPs_50, exp_GS_MEPs_25, exp_GS_EMG_onsets_three_stim), method='Nelder-Mead', options={'maxiter':1500,'ftol':0.1})    
    print "ParamsOptimizedGS", optGS
    params_GS = optGS.x
    return params_GS
#%%
if __name__ == "__main__":
    if sys.argv[1] == "ATM_first":
        build_Go_GS_ATM()
    elif sys.argv[1] == "ATM_second":
        build_GS_facNew()
    elif sys.argv[1] == "HRM":
        build_Go_GS_HRM()
    else:
        print "Unrecognized argument"
    
#%%