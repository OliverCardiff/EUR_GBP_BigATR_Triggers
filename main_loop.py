# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 09:15:51 2019

@author: Oliver
"""
runfile('read_data.py')
runfile('indicator_functs.py')
runfile('portfolio.py')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import pickle as pkl
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

#This global variable will be used to define a basic bid/ask spread
#Based on retail trading platforms offering fixed spreads 1.5-2.5 for these pairs
pip_penalty = 0.0001 # will be applied twice per trade (Op/Cl)

#Optimisation Parameters for Triggers ((not longer optimised!))
ATR_Relative = 1.136 #ATR20 as scale for 'Big Move' detection 
Trade_window = 4 #Time window to model characteristics of

#Indicator variables again for reference (not optimising these)
atr_scale = 20
MA_slow = 150
MA_fast = 40
rsi_period = 14

#These are defined outside the function to make the bayesian opt easier
current_interval_index = 1 #refer to array below = 15min, 1hr, 4hr, 1D

#establishing the train/test split and indexing
intervals = [15,60,240,1440]
##two way train test split (at 0.39 to avoid a weird price-shock error)
train = int(eu_data[current_interval_index].shape[0] * 0.39)

##three way train test split
train1 = int(eu_data[current_interval_index].shape[0] * 0.333)
train2 = train1 * 2

#time interval variable with will be used later
glob_interval = intervals[current_interval_index]

#This global variable (list of two) will contain the min/max models
res = pkl.load(open('picks/4H_models.pickle', "rb"))
#This global variable is a reference to the portfolio instance
port = Portfolio(10000.0, eu_data[3].shape[0])


"""
pkl.dump(obj=res, file=open("1D_models.pickle", "wb"))
pkl.dump(obj=optim_BackTest, file=open("1D_backtest.pickle", "wb"))
pkl.dump(obj=optim_Trigger, file=open("1D_trigger.pickle", "wb"))
pkl.dump(obj=port.perfList, file=open("1D_perfList.pickle", "wb"))
pkl.dump(obj=port.days, file=open("1D_days.pickle", "wb"))

pkl.dump(obj=res, file=open("4H_models.pickle", "wb"))
pkl.dump(obj=optim_BackTest, file=open("4H_backtest.pickle", "wb"))
pkl.dump(obj=optim_Trigger, file=open("4H_trigger.pickle", "wb"))
pkl.dump(obj=port.perfList, file=open("4H_perfList.pickle", "wb"))
pkl.dump(obj=port.days, file=open("4H_days.pickle", "wb"))
"""

"""
No longer three way split, just two!
dfeu = eu_data[current_interval_index][:train1]
dfgb = gb_data[current_interval_index][:train1]

"""

"""
No Longer Optimising the triggers - doesn't work
#res = []
trig_bounds = {'ATRsc' : (0.8,1), 'twind' : (4,4)}

optim_Trigger = BayesianOptimization(
        f=TriggerTrain,
        pbounds=trig_bounds,
        verbose=2,
        random_state=4)

optim_Trigger.maximize(init_points=3, n_iter=8)

ATR_Relative = optim_Trigger.max["params"]["ATRsc"]
Trade_window = optim_Trigger.max["params"]["twind"]
"""

"""
No longer three way split, just two!
dfeu = eu_data[current_interval_index][train1:train2]
dfgb = gb_data[current_interval_index][train1:train2]

"""

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vvvvvvvvvv AREA WHERE ALL FUNCT ARE RAN AND OPTS PERFORMED ETC. vvvvvvvvvvvv 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
ATR_Relative = 1
Trade_window = 3
rr = 1.2
nmem2 = ParaMemory(names=['ATRsc1', 'Twindow', 'Risk_ratio', 'Qprofit', 'Qloss',
                         'ATRsc2', 'MSE', 'profit'], days=eu_data[3].shape[0])

bounds = {'Para1' : (1.15,1.25), 'Para2' : (18,22),  'Para3' : (1.2,1.3)}

optim_paras = BayesianOptimization(
            f=partial(SearchForParams, tmem=nmem2),
            pbounds=bounds,
            verbose=2,
            random_state=np.random.randint(50,10000))

optim_paras.maximize(init_points=10, n_iter=20)

SearchForParams(ATR_Relative, Trade_window, rr, nmem2, verb=2)

plotTstTrain(nmem2.folios[4].perfList, nmem2.folios[4].days, "1H_Train_Test")


def SearchForParams(Para1, Para2, Para3, tmem, verb=0):
    global dfeu
    global dfgb
    dfeu = eu_data[current_interval_index][:train]
    dfgb = gb_data[current_interval_index][:train]
    
    res = TriggerTrain(Para1, Para2)
    nmse = np.mean([res[0].best_score_, res[1].best_score_])
    if(verb > 0):
        print("MSE: " + str(nmse))
    
    tmem.PS.D[0] = Para1
    tmem.PS.D[1] = Para2
    tmem.PS.D[2] = Para3
    
    #Setting optimisation ranges for trade system params
    back_bounds = {'Qprofit' : (0.8,1.3), 'Qloss' : (0.7,1.3), 
                   'ATRsc' : (Para1-0.2,Para1+0.2)}
    
    optim_BackTest = BayesianOptimization(
            f=partial(RunBackTest, twind=Para2, Risk_ratio = Para3,
                      port=tmem.PF,ests=res),
            pbounds=back_bounds,
            verbose=verb,
            random_state=np.random.randint(50,10000))
    
    optim_BackTest.maximize(init_points=6, n_iter=8)
    
    tmem.PS.D[3] = optim_BackTest.max["params"]["Qprofit"]
    tmem.PS.D[4] = optim_BackTest.max["params"]["Qloss"]
    tmem.PS.D[5] = optim_BackTest.max["params"]["ATRsc"]
    
    dfeu = eu_data[current_interval_index][int(train*1.03):]
    dfgb = gb_data[current_interval_index][int(train*1.03):]
    
    prof = RunBackTest(tmem.PS.D[0],tmem.PS.D[1], tmem.PS.D[3], tmem.PS.D[4],
                       Risk_ratio = tmem.PS.D[2], port=tmem.PF, ests=res)
    
    tmem.PS.D[6] = nmse
    tmem.PS.D[7] = prof
    
    tmem.NewRun()
    
    return prof

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
^^^^^^^^^^^^^^^^^^^^ END OF MESSY EXECUTION AREA OF SCRIPT ^^^^^^^^^^^^^^^^^^ 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


"""
This function is designed to be black-box optimised. The idea is to tune
the trade triggers for model predictive power BEFORE optimising trading behaviour
This function should be optimised on the first third of a three-way split
"""
def TriggerTrain(ATRsc, twind):
   
    twind = int(twind)
    #Run the trigger generation function
    sinds = SetTriggers(ATRsc, twind)
    #Run the predictive feature building function
    featdep = BuildFeatures(sinds, twind)
    
    #Building the RandomForestRegressor here
    
    #Set up an initial search space for hyperparameters
    md = np.linspace(3,8,6).astype(int)
    nest = np.linspace(20,100,4).astype(int)
    feat = np.arange(3,7,1)
    tuned_params = [{'max_depth': md, 'n_estimators': nest,
                     'max_features' : feat}]
    
    #Use a time-slice CV because this is a time series!
    my_cv1 = TimeSeriesSplit(n_splits=4)
    my_cv2 = TimeSeriesSplit(n_splits=4)
    
    #Grid Search for best params to predict both high and low price over time window
    clfH = GridSearchCV(RandomForestRegressor(n_estimators=30, min_samples_split=20), tuned_params,
                       cv=my_cv1, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    clfL = GridSearchCV(RandomForestRegressor(n_estimators=30, min_samples_split=20), tuned_params,
                       cv=my_cv2, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    #fit the final model data
    clfL.fit(featdep[0], featdep[1]['MINP'])
    clfH.fit(featdep[0], featdep[1]['MAXP'])
    
    #return the maximisable mean of both NMSE
    return [clfL, clfH]


"""
This function is designed to be black-box optimised too. The objective is to
optimise the trading paramters which goven how the algorithm sets its stops
relative to the estimates provided by the RandomForest, and their risk.
This function should be optimised on the same training data as the RF.
"""
def RunBackTest(ATRsc=ATR_Relative, twind=Trade_window, Qprofit=0.8, Qloss=1.2, Risk_ratio=1.2,
                lotsize=0.01, leverage=50, minstop=0.001, maxstop=0.1, port=None, ests=None):
    #Calls the portfolio reset function, portfolio is passed by reference
    #and persists between Backtests, resets save last performance in a list
    port.ResetIt()
    global res
    #Call the signal generation function, now with optimised parameters
    sinds = SetTriggers(ATRsc, twind)
    #call the feature building function
    featdep = BuildFeatures(sinds, twind)
    
    #if for some reason the model don't trade, exit with 0 profit
    if(featdep[0].shape[0] < 1):
        return port.GetFinalReturn()
    
    #generate low and high price estimates from features using randforestregressors
    l_est = ests[0].predict(featdep[0])
    h_est = ests[1].predict(featdep[0])
    
    #set the low differentials as positive
    l_est *= -1
    
    #create high and low stop-losses for longs and shorts
    hscaleLoss = (h_est * Qloss)
    lscaleLoss = (l_est * Qloss)
    
    #create high and low profit-targets for longs and shorts
    hscaleProf = (h_est * Qprofit)
    lscaleProf = (l_est * Qprofit)
    
    #Create long and short risk ratios modified by optimisable paramters
    Lratio = np.log((hscaleProf)/(lscaleLoss))
    Sratio = np.log((lscaleProf)/(hscaleLoss))
    #log the risk ratio for comparison
    Qratio = np.log(Risk_ratio)
    
    #create a series of entry prices for each trade (the trigger closes)
    entryPrice = gb_data[current_interval_index].iloc[np.asarray(sinds, dtype='int')]['CL']
    entryPrice.index = pd.RangeIndex(len(sinds))
    
    #create high and low stop-losses for longs and shorts
    hstops = entryPrice + (hscaleLoss)
    lstops = entryPrice - (lscaleLoss)
    
    #create high and low profit-targets for longs and shorts
    hprofs = entryPrice + (hscaleProf)
    lprofs = entryPrice - (lscaleProf)
    
    #The main trade execution loop
    for i in range(len(sinds)):
        posSize = (port.GetCurrent(sinds[i], glob_interval) * lotsize) * leverage
        delta = 0
        #check risk ratio is acceptable, and stop is small/large enough
        if((Qratio < Lratio[i]) & (lscaleLoss[i] > minstop) & (lscaleLoss[i] < maxstop)):
            #longs are executed a pip higher than the quote
            ePrice = entryPrice[i] + pip_penalty
            #Simulate trade progression at minute leve resolution
            rets = ScanMinData(sinds[i], twind, hprofs[i], lstops[i])
            #Long-closes are executed a pip lower than the stop
            exitPrice = rets[0] - pip_penalty
            #find difference in position value as result of trade
            delta = (exitPrice - ePrice) * posSize
        #check risk ratio is acceptable, and stop is small/large enough
        elif((Qratio < Sratio[i]) & (hscaleLoss[i] > minstop) & (hscaleLoss[i] < maxstop)):
            #Shorts are executed a pip lower than the quote
            ePrice = entryPrice[i] - pip_penalty
            #Simulate trade progression at minute leve resolution
            rets = ScanMinData(sinds[i], twind, hstops[i], lprofs[i])
            #Short-closes are executed a pip higher than the stop
            exitPrice = rets[0] + pip_penalty
            #find difference in position value as result of trade
            delta = (ePrice - exitPrice) * posSize
            
        if(abs(delta) > 0):
            port.AddDelta(sinds[i], delta, glob_interval)
        
    return port.GetFinalReturn()
    
"""
This function applies the fairly simple logic for the trade triggers:
If movement > ATR*scalingFactor, try to trade!
"""
def SetTriggers(Atrsc, twind, mod):
    #create a signals dataframe on the same time-index
    signal = pd.DataFrame(index=dfeu.index)
    #modify the signal for up/down moves
    difftest = (dfeu['OP'] - dfeu['CL']) * mod
    #run the trade entry logic
    signal['BIG_MOVE'] = np.where(Atrsc * dfeu['ATR']
          < difftest, 1,0)
    
    #save the indexes as a column
    signal['INDS'] = signal.index
    term = signal['INDS'].max()
    #select all time-indexes corresponding to a trade signal above the min
    #index for the slow MA and sufficiently far away from the end of time
    sind = signal.loc[(signal['BIG_MOVE'] == 1) & (signal['INDS'] > MA_slow)
           & (signal['INDS'] < (term - (twind*2)))]
    sinds = sind['INDS'].tolist()
    
    #return a list of time indexes to trade at
    return sinds

"""
Although the trading strategy may be triggered by different size time
intervals, all trades are simulated with the minute-level data (the best
resolution available here)  
"""
def ScanMinData(ind, twind, hstop, lstop):
    #start scanning per-minute at the start of interval after the trigger
    #the trigger interval index being 'ind', multiplied by the global for minutes
    twind = int(twind)
    expI = (ind+1) * glob_interval
    #set converted time window limit
    lim = expI + (glob_interval * twind)
    
    ##get close price on time window
    tcl = gbps.iloc[lim]['<CLOSE>']
    
    #Sanity/error check
    if(tcl > hstop):
        tcl = hstop
    if(tcl < lstop):
        tcl = lstop
    
    #get set of 1min highs and lows over window
    mx = gbps.iloc[expI:lim:1]['<HIGH>']
    mn = gbps.iloc[expI:lim:1]['<LOW>']
    
    #if trade isn't price-stopped it gets time-stopped at the close price
    fin = tcl
    indTrack = expI
    for i in mx.index:
        #check if price moves above or below stops during window, if so, end trade
        if(mx[i] >= hstop):
            fin = hstop
            indTrack = i
            break
        if(mn[i] <= lstop):
            fin = lstop
            indTrack = i
            break
    #returning the final trade close price and its time index
    return [fin, indTrack/glob_interval]

"""
This function is for building features! It collects a set of features per
trade to be executed (per row). Features are indexed sequentially, but 
correspond in order to the time-indexes in the sinds list
"""
def BuildFeatures(sinds, twind):
    cls = ['ATR_SC', 'GBP_PROP', 'GBMAS', 'GBMAF', 'EUMAS', 'EUMAF', 'GBRSI', 'EURSI', 'EUATR', 'GBATR']
    cl2 = ['MINP', 'MAXP']
    features = pd.DataFrame(index=pd.RangeIndex(len(sinds)), columns=cls, data=0.0)
    minmaxPrice = pd.DataFrame(index=pd.RangeIndex(len(sinds)), columns=cl2, data=0.0)

    for i in range(len(sinds)):
        ind = sinds[i]
        nextInd = sinds[i] + 1
        limitInd = nextInd + twind
        
        euop = dfeu.loc[ind]['OP']
        eucl = dfeu.loc[ind]['CL']
        gbop = dfgb.loc[ind]['OP']
        gbcl = dfgb.loc[ind]['CL']
        
        #Get the dependent vars (min and max price in time-window)
        high = dfgb.loc[nextInd:limitInd:1]['HI'].max()
        low = dfgb.loc[nextInd:limitInd:1]['LO'].min()
        minmaxPrice.loc[i]['MINP'] = low - gbcl
        minmaxPrice.loc[i]['MAXP'] = high - gbcl
        
        #Calculate the features
        prp1 = (euop - eucl) / euop
        prp2 = (gbop - gbcl) / gbop
        
        features.iloc[i]['ATR_SC'] = np.log(abs(euop - eucl) / dfeu.loc[ind]['ATR'])
        features.iloc[i]['GBP_PROP'] = prp2/prp1;
        features.iloc[i]['GBMAS'] = np.log(gbcl / dfgb.loc[ind]['MAS'])
        features.iloc[i]['GBMAF'] = np.log(gbcl / dfgb.loc[ind]['MAF'])
        features.iloc[i]['EUMAS'] = np.log(eucl / dfeu.loc[ind]['MAS'])
        features.iloc[i]['EUMAF'] = np.log(eucl / dfeu.loc[ind]['MAF'])
        features.iloc[i]['GBRSI'] = dfgb.loc[ind]['RSI']
        features.iloc[i]['EURSI'] = dfeu.loc[ind]['RSI']
        features.iloc[i]['EUATR'] = dfeu.loc[ind]['ATR']
        features.iloc[i]['GBATR'] = dfgb.loc[ind]['ATR']
    
    return [features, minmaxPrice]
