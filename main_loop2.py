# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:00:30 2019

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

import seaborn as sns

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
current_interval_index = 2 #refer to array below = 15min, 1hr, 4hr, 1D

#establishing the train/test split and indexing
intervals = [15,60,240,1440]
##two way train test split (at 0.39 to avoid a weird price-shock error)
train = int(eu_data[current_interval_index].shape[0] * 0.39)

##three way train test split
#train1 = int(eu_data[current_interval_index].shape[0] * 0.333)
#train2 = train1 * 2

#time interval variable with will be used later
glob_interval = intervals[current_interval_index]

#This global variable (list of two) will contain the min/max models
#res = pkl.load(open('picks/4H_models.pickle', "rb"))
#This global variable is a reference to the portfolio instance
port = Portfolio(10000.0, eu_data[3].shape[0])



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vvvvvvvvvv AREA WHERE ALL FUNCT ARE RAN AND OPTS PERFORMED ETC. vvvvvvvvvvvv 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

nmem2 = ParaMemory(names=['ATRscH', 'ATRscL', 'Twindow', 'Risk_ratio', 'Qprofit', 'Qloss',
                         'ATRscH2', 'ATRscL2', 'MSE', 'profit'], days=eu_data[3].shape[0])

bounds = {'Para1' : (0.8,1.3), 'Para2' : (0.8,1.3), 'Para3' : (4,10),  'Para4' : (1.05,1.3)}

optim_paras = BayesianOptimization(
            f=partial(SearchForParams, tmem=nmem2),
            pbounds=bounds,
            verbose=2,
            random_state=np.random.randint(50,10000))

optim_paras.maximize(init_points=10, n_iter=20)

SearchForParams(ATR_Relative, Trade_window, rr, nmem2, verb=2)

plotTstTrain(nmem2.folios[4].perfList, nmem2.folios[4].days, "1H_Train_Test")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
^^^^^^^^^^^^^^^^^^^^ END OF MESSY EXECUTION AREA OF SCRIPT ^^^^^^^^^^^^^^^^^^ 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

ATR_R1 = 1
ATR_R2 = 1
Trade_window = 8
rr = 1.30

dfeu = eu_data[current_interval_index][:train]
dfgb = gb_data[current_interval_index][:train]

res2 = TriggerTrain2(ATR_R1, ATR_R2, Trade_window, 0.96, 8)
nmse = [res2[0].best_score_, res2[1].best_score_,
                res2[2].best_score_, res2[3].best_score_]
nmse
if(verb > 0):
    print("MSE: " + str(nmse))


port = Portfolio(10000.0, eu_data[3].shape[0])
#Setting optimisation ranges for trade system params
back_bounds = {'Qprofit' : (0.9,1.6), 'Qloss' : (0.7,1.1), 
               'ATRscH' : (0.75,1.3), 'ATRscL' : (0.85,1.2)}

optim_BackTest = BayesianOptimization(
        f=partial(RunBackTest, twind=Trade_window, Risk_ratio = rr,
                  port=port,ests=res2),
        pbounds=back_bounds,
        verbose=2,
        random_state=np.random.randint(50,10000))

optim_BackTest.maximize(init_points=8, n_iter=8)

plotTstTrain(port.perfList, port.days, "Delete_me")

qprof = optim_BackTest.max["params"]["Qprofit"]
qloss = optim_BackTest.max["params"]["Qloss"]
ATR_R1 = optim_BackTest.max["params"]["ATRscH"]
ATR_R2 = optim_BackTest.max["params"]["ATRscL"]

dfeu = eu_data[current_interval_index][int(train*1.03):]
dfgb = gb_data[current_interval_index][int(train*1.03):]


prof = RunBackTest(ATR_R1,ATR_R2, qprof, qloss,
                   Risk_ratio = rr, port=port, ests=res2)
prof

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
^^^^^^^^^^^^^^^^^^^^ END OF MESSY EXECUTION AREA OF SCRIPT ^^^^^^^^^^^^^^^^^^ 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

"""
This functions combines the trigger training with the backtest optimisation
and the final optimised test set validation
"""
def SearchForParams(Para1, Para2, Para3, Para4, tmem, verb=0):
    global dfeu
    global dfgb
    dfeu = eu_data[current_interval_index][:train]
    dfgb = gb_data[current_interval_index][:train]
    
    res = TriggerTrain2(Para1, Para2, Para3)
    nmse = np.mean([res[0].best_score_, res[1].best_score_,
                    res[2].best_score_, res[3].best_score_])
    if(verb > 0):
        print("MSE: " + str(nmse))
    
    tmem.PS.D[0] = Para1
    tmem.PS.D[1] = Para2
    tmem.PS.D[2] = Para3
    tmem.PS.D[3] = Para4
    
    #Setting optimisation ranges for trade system params
    back_bounds = {'Qprofit' : (0.9,1.3), 'Qloss' : (0.7,1.1), 
                   'ATRscH' : (Para1-0.2,Para1+0.2), 'ATRscL' : (Para2-0.2,Para2+0.2)}
    
    optim_BackTest = BayesianOptimization(
            f=partial(RunBackTest, twind=Para2, Risk_ratio = Para3,
                      port=tmem.PF,ests=res),
            pbounds=back_bounds,
            verbose=verb,
            random_state=np.random.randint(50,10000))
    
    optim_BackTest.maximize(init_points=6, n_iter=8)
    
    tmem.PS.D[3] = optim_BackTest.max["params"]["Qprofit"]
    tmem.PS.D[4] = optim_BackTest.max["params"]["Qloss"]
    tmem.PS.D[5] = optim_BackTest.max["params"]["ATRscH"]
    tmem.PS.D[6] = optim_BackTest.max["params"]["ATRscL"]
    
    dfeu = eu_data[current_interval_index][int(train*1.03):]
    dfgb = gb_data[current_interval_index][int(train*1.03):]
    
    prof = RunBackTest(tmem.PS.D[0],tmem.PS.D[1], tmem.PS.D[3], tmem.PS.D[4],
                       Risk_ratio = tmem.PS.D[2], port=tmem.PF, ests=res)
    
    tmem.PS.D[7] = nmse
    tmem.PS.D[8] = prof
    
    tmem.NewRun()
    
    return prof

"""
This function is designed to be black-box optimised. The idea is to tune
the trade triggers for model predictive power BEFORE optimising trading behaviour
"""

def TriggerTrain2(ATRscH, ATRscL, twind, lr, ne):
   
    twind = int(twind)
    #Run the trigger generation function
    mult = MultiTrig(ATRscH, ATRscL, twind)
    
    #Building the RandomForestRegressor here
    
    #Set up an initial search space for hyperparameters
    params = {
        'min_child_weight': [60,80,100],
        'gamma': [0],
        'subsample': [0.5,0.65],
        'colsample_bytree': [0.7, 0.8],
        'max_depth': [3],
        'reg_lambda': [0.5,0.7,1]
        }
    
    xgb = XGBRegressor(learning_rate=lr, n_estimators=ne, objective='reg:linear',
                       silent=True, nthread=1)
    
    #Use a time-slice CV because this is a time series!
    my_cv1 = TimeSeriesSplit(n_splits=4)
    my_cv2 = TimeSeriesSplit(n_splits=4)
    my_cv3 = TimeSeriesSplit(n_splits=4)
    my_cv4 = TimeSeriesSplit(n_splits=4)
   
    #Grid Search for best params to predict both high and low price over time window
    HH = GridSearchCV(xgb, params,
                       cv=my_cv1, scoring='r2', verbose=0, n_jobs=-1)
    HL = GridSearchCV(xgb, params,
                       cv=my_cv2, scoring='r2', verbose=0, n_jobs=-1)
    LH = GridSearchCV(xgb, params,
                       cv=my_cv3, scoring='r2', verbose=0, n_jobs=-1)
    LL = GridSearchCV(xgb, params,
                       cv=my_cv4, scoring='r2', verbose=0, n_jobs=-1)
    
    #print(mult[0][1].head())
    #fit the final model data
    HH.fit(mult[0][0], mult[0][1]['MAXP'])
    HL.fit(mult[0][0], mult[0][1]['MINP'])
    LH.fit(mult[1][0], mult[1][1]['MAXP'])
    LL.fit(mult[1][0], mult[1][1]['MINP'])
    
    #outs = HH.predict(mult[0][0])
    #outs1 = HL.predict(mult[0][0])
   # outs2 = LH.predict(mult[1][0])
   # outs3 = LL.predict(mult[1][0])
    
   # print(str(outs))
   # print(str(outs1))
   # print(str(outs2[:30]))
   # print(str(outs3[:30]))
    
    #return the maximisable mean of both NMSE
    return [HH, HL, LH, LL]

def MultiTrig(ATRscH, ATRscL, twind):
    sH = SetTriggers(ATRscH, twind, 1)
    sL = SetTriggers(ATRscL, twind, -1)
    
    fH = BuildFeatures(sH, twind)
    fL = BuildFeatures(sL, twind)
    
    return [fH, fL, sH, sL]
    

def MakeMat(feat, sind, Qloss, Qprofit, est1, est2):
    #generate low and high price estimates from features using randforestregressors
    h_est = est1.predict(feat)
    l_est = est2.predict(feat)
    
    #for i in range(int(len(h_est)/10)):
    #    print(str(h_est[i]) + " " + str(l_est[i]))
    
    #set the low differentials as positive
    l_est *= -1
    
    #create high and low stop-losses for longs and shorts
    hscaleLoss = (h_est * Qloss)
    lscaleLoss = (l_est * Qloss)
    
    #create high and low profit-targets for longs and shorts
    hscaleProf = (h_est * Qprofit)
    lscaleProf = (l_est * Qprofit)
    
    #Create long and short risk ratios modified by optimisable paramters
    Lratio = (hscaleProf)/(lscaleLoss)
    Sratio = (lscaleProf)/(hscaleLoss)
    
    #create a series of entry prices for each trade (the trigger closes)
    entryPrice = gb_data[current_interval_index].loc[np.asarray(sind, dtype='int')]['CL']
    entryPrice.index = pd.RangeIndex(len(sind))
    
    #create high and low stop-losses for longs and shorts
    hstops = entryPrice + (hscaleLoss)
    lstops = entryPrice - (lscaleLoss)
    
    #create high and low profit-targets for longs and shorts
    hprofs = entryPrice + (hscaleProf)
    lprofs = entryPrice - (lscaleProf)
    
    d = {'sinds' : sind, 'HSLoss' : hscaleLoss, 'LSLoss' : lscaleLoss, 
         'Lratio': Lratio, 'Sratio' : Sratio, 'hstops' : hstops,
         'lstops' : lstops, 'hprofs' : hprofs, 'lprofs' : lprofs,
         'entry' : entryPrice}
    
    return pd.DataFrame(d)

"""
This function is designed to be black-box optimised too. The objective is to
optimise the trading paramters which goven how the algorithm sets its stops
relative to the estimates provided by the RandomForest, and their risk.
This function should be optimised on the same training data as the RF.
"""
def RunBackTest(ATRscH=1, ATRscL=1, twind=1, Qprofit=0.8, Qloss=1.2, Risk_ratio=1.2,
                lotsize=0.01, leverage=50, minstop=0.001, maxstop=0.1, port=None, ests=None):
    #Calls the portfolio reset function, portfolio is passed by reference
    #and persists between Backtests, resets save last performance in a list
    port.ResetIt()
    #Call the signal generation function, now with optimised parameters
    mult = MultiTrig(ATRscH, ATRscL, twind)

    #Run estimators and make prof/loss matrix
    df1 = MakeMat(mult[0][0], mult[2], Qloss, Qprofit, ests[0], ests[1])
    df2 = MakeMat(mult[1][0], mult[3], Qloss, Qprofit, ests[2], ests[3])
    
    #merge two matrices
    dfx = pd.concat([df1, df2], join='outer')
    
    #index them by their time-index
    dfx.set_index('sinds', drop=False, inplace=True)
    #sort by time-index to get a consecutive series of trades
    dfx.sort_index(inplace=True)
    
    #log the risk ratio for comparison
    Qratio = Risk_ratio
    
    print("This Many Ops: " + str(len(dfx.index)))
    cnt = 0
    
    #The main trade execution loop
    for i,r in dfx.iterrows():
        posSize = (port.GetCurrent(r['sinds'], glob_interval) * lotsize) * leverage
        delta = 0
        #if i % 15 == 0:
        #        print(str(i) + "\n" + str(r))
        #print(str(i) + " " + str(r))
        #check risk ratio is acceptable, and stop is small/large enough
        if((Qratio < r['Lratio']) & (r['LSLoss'] > minstop) & (r['LSLoss'] < maxstop)):
            cnt += 1
            #longs are executed a pip higher than the quote
            ePrice = r['entry'] + pip_penalty
            #Simulate trade progression at minute level resolution
            rets = ScanMinData(r['sinds'], twind, r['hprofs'], r['lstops'])
            #Long-closes are executed a pip lower than the stop
            exitPrice = rets[0] - pip_penalty
            #find difference in position value as result of trade
            delta = (exitPrice - ePrice) * posSize
        #check risk ratio is acceptable, and stop is small/large enough
        if((Qratio < r['Sratio']) & (r['HSLoss'] > minstop) & (r['HSLoss'] < maxstop)):
            cnt += 1
            #Shorts are executed a pip lower than the quote
            ePrice = r['entry'] - pip_penalty
            #Simulate trade progression at minute level resolution
            rets = ScanMinData(r['sinds'], twind, r['hstops'], r['lprofs'])
            #Short-closes are executed a pip higher than the stop
            exitPrice = rets[0] + pip_penalty
            #find difference in position value as result of trade
            delta = (ePrice - exitPrice) * posSize
            
        if(abs(delta) > 0):
            port.AddDelta(r['sinds'], delta, glob_interval)
    
    print("This Many Trades: " + str(cnt)) 
    #if for some reason the model don't trade, exit with 0 profit
    if(cnt < int(dfx.shape[0]/5)):
        return port.GetFinalReturn() + (-200 * (1 - cnt/(dfx.shape[0]/5)))
     
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
    tcl = gbps.loc[lim]['<CLOSE>']
    
    #Sanity/error check
    if(tcl > hstop):
        tcl = hstop
    if(tcl < lstop):
        tcl = lstop
    
    #get set of 1min highs and lows over window
    mx = gbps.loc[expI:lim:1]['<HIGH>']
    mn = gbps.loc[expI:lim:1]['<LOW>']
    
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
        
        #if i % 15 == 0:
        #    print(str(minmaxPrice.loc[i]['MINP']) + " " + str(minmaxPrice.loc[i]['MAXP']))
        
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
