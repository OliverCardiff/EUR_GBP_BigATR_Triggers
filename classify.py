# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:37:45 2019

@author: Oliver
"""

import numpy as np
import pandas as pd
import seaborn as sns

from functools import partial

from bayes_opt import BayesianOptimization as BOpt

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb

current_interval_index = 2
intervals = [15,60,240,1440]

train = int(eu_data[current_interval_index].shape[0] * 0.39)

glob_interval = intervals[current_interval_index]

port = Portfolio(10000.0, eu_data[3].shape[0])

dfeu = eu_data[current_interval_index][:train]
dfgb = gb_data[current_interval_index][:train]

dfeu = eu_data[current_interval_index][int(train*1.03):]
dfgb = gb_data[current_interval_index][int(train*1.03):]

mult = MultiTrig(1.2,1.2,6)

mtest = MultiTrig(1.2,1.2,6)

rr = 1.3

do_longH = pd.Series(np.where((abs(mult[0][1]['MINP']) * rr) < abs(mult[0][1]['MAXP']), 1, 0), name='longs') 
do_longL = pd.Series(np.where((abs(mult[1][1]['MINP']) * rr) < abs(mult[1][1]['MAXP']), 1, 0), name='longs')
do_shortH = pd.Series(np.where((abs(mult[0][1]['MAXP']) * rr) < abs(mult[0][1]['MINP']), 1, 0), name='shorts')
do_shortL = pd.Series(np.where((abs(mult[1][1]['MAXP']) * rr) < abs(mult[1][1]['MINP']), 1, 0), name='shorts')

Tdo_longH = pd.Series(np.where((abs(mtest[0][1]['MINP']) * rr) < abs(mtest[0][1]['MAXP']), 1, 0), name='longs') 
Tdo_longL = pd.Series(np.where((abs(mtest[1][1]['MINP']) * rr) < abs(mtest[1][1]['MAXP']), 1, 0), name='longs')
Tdo_shortH = pd.Series(np.where((abs(mtest[0][1]['MAXP']) * rr) < abs(mtest[0][1]['MINP']), 1, 0), name='shorts')
Tdo_shortL = pd.Series(np.where((abs(mtest[1][1]['MAXP']) * rr) < abs(mtest[1][1]['MINP']), 1, 0), name='shorts')

"""
###############################################################################
vv feature transform vv
###############################################################################
"""

cls = mult[0][0].columns
ssH = StandardScaler().fit(mult[0][0])
featH = pd.DataFrame(ssH.transform(mult[0][0]))
featH.columns = cls
pcaH = PCA()
pcaH.fit(featH)
featH = pd.DataFrame(pcaL.transform(featH))

ssL = StandardScaler().fit(mult[1][0])
featL = pd.DataFrame(ssL.transform(mult[1][0]))
featL.columns = cls
pcaL = PCA()
pcaL.fit(featL)
featL = pd.DataFrame(pcaL.transform(featL))

TfeatH = pd.DataFrame(ssH.transform(mtest[0][0]))
TfeatL = pd.DataFrame(ssL.transform(mtest[1][0]))
TfeatH = pd.DataFrame(pcaH.transform(TfeatH))
TfeatL = pd.DataFrame(pcaL.transform(TfeatL))

"""
###############################################################################
vv plotting vv
###############################################################################
"""

data = pd.concat([do_shortH, featH], axis=1)
data = pd.melt(data, id_vars='shorts', var_name='features')
plt.figure()
sns.violinplot(x='features', y='value', hue='shorts', data=data, split=True, inner='quart')
plt.xticks(rotation=90)
plt.show()

data = pd.concat([do_longL, featL], axis=1)
data = pd.melt(data, id_vars='longs', var_name='features')
plt.figure()
sns.violinplot(x='features', y='value', hue='longs', data=data, split=True, inner='quart')
plt.xticks(rotation=90)
plt.show()

g = sns.PairGrid(featL, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)
g.map_upper(plt.scatter)

"""
###############################################################################
vv model building vv
###############################################################################
"""
matHH = xgb.DMatrix(featH, do_longH)
matHL = xgb.DMatrix(featH, do_shortH)
matLH = xgb.DMatrix(featL, do_longL)
matLL = xgb.DMatrix(featL, do_shortL)
    
paras = {'learning_rate': (0.002,0.5), 'max_depth' : (2,11),
         'min_child_weight': (10,60), 'colsample_bytree' : (0.7,1),
         'gamma' : (0,0.1), 'subsample' : (0.7,1)}


def OptXGB(train, learning_rate, max_depth, min_child_weight, gamma, subsample, colsample_bytree):
    xg = xgb.XGBClassifier(n_estimators=200, learning_rate=learning_rate,
                           max_depth=int(max_depth), min_child_weight=min_child_weight,
                           gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree,
                           scale_pos_weight=1)
    xgpara = xg.get_xgb_params()
    cv_res = xgb.cv(xgpara, train, metrics=['map'], nfold=8)
    return cv_res['test-map-mean'].iloc[-1]

def runXGB(train, ys, para):
    xg = xgb.XGBClassifier(n_estimators=200,
                           learning_rate=para['learning_rate'],
                           max_depth=int(para['max_depth']),
                           min_child_weight=para['min_child_weight'],
                           gamma=para['gamma'], 
                           subsample=para['subsample'],
                           colsample_bytree=para['colsample_bytree'],
                           scale_pos_weight=11)
    xg.fit(train, ys)
    return xg

boHH = BOpt(partial(OptXGB, train=matHH),
            random_state=np.random.randint(50,10000),
            pbounds=paras)
boHL = BOpt(partial(OptXGB, train=matHL),
            random_state=np.random.randint(50,10000),
            pbounds=paras)
boLH = BOpt(partial(OptXGB, train=matLH),
            random_state=np.random.randint(50,10000),
            pbounds=paras)
boLL = BOpt(partial(OptXGB, train=matLL),
            random_state=np.random.randint(50,10000),
            pbounds=paras)

boHH.maximize(init_points=100, n_iter=30)
boHL.maximize(init_points=100, n_iter=30)
boLH.maximize(init_points=100, n_iter=30)
boLL.maximize(init_points=100, n_iter=30)

xgHH = runXGB(featH, do_longH, boHH.max["params"])
xgHL = runXGB(featH, do_shortH, boHL.max["params"])
xgLH = runXGB(featL, do_longL, boLH.max["params"])
xgLL = runXGB(featL, do_shortL, boLL.max["params"])

    

"""
###############################################################################
vv feature transform vv
###############################################################################
"""

gnbHH = GaussianNB()
gnbHL = GaussianNB()
gnbLH = GaussianNB()
gnbLL = GaussianNB()

gnbHH.fit(featH, do_longH)
gnbHL.fit(featH, do_shortH)
gnbLH.fit(featL, do_longL)
gnbLL.fit(featL, do_shortL)

rf = RFC(n_estimators=100)
my_cv = RepeatedKFold(n_splits=5, n_repeats=10)
params = {'max_depth' : (3,4,5,6), 'min_samples_split' : (20,30,40,50,60,70)}

clfHH = GridSearchCV(rf, params, cv=my_cv, scoring='precision', n_jobs=-1)
clfHL = GridSearchCV(rf, params, cv=my_cv, scoring='precision', n_jobs=-1)
clfLH = GridSearchCV(rf, params, cv=my_cv, scoring='precision', n_jobs=-1)
clfLL = GridSearchCV(rf, params, cv=my_cv, scoring='precision', n_jobs=-1)

clfHH.fit(featH, do_longH)
clfHL.fit(featH, do_shortH)
clfLH.fit(featL, do_longL)
clfLL.fit(featL, do_shortL)


"""
###############################################################################
vv RFC Validation vv
###############################################################################
"""

CheckTradeRatio(clfHH, featH, TfeatH, do_longH, Tdo_longH)
CheckTradeRatio(clfHL, featH, TfeatH, do_shortH, Tdo_shortH)
CheckTradeRatio(clfLH, featL, TfeatL, do_longL, Tdo_longL)
CheckTradeRatio(clfLL, featL, TfeatL, do_shortL, Tdo_shortL)

"""
###############################################################################
vv XGB Validation vv
###############################################################################
"""

CheckTradeRatio(xgHH, featH, TfeatH, do_longH, Tdo_longH)
CheckTradeRatio(xgHL, featH, TfeatH, do_shortH, Tdo_shortH)
CheckTradeRatio(xgLH, featL, TfeatL, do_longL, Tdo_longL)
CheckTradeRatio(xgLL, featL, TfeatL, do_shortL, Tdo_shortL)

"""
###############################################################################
^^ RFC Validation ^^
vv GNB Validation vv
###############################################################################
"""

CheckTradeRatio(gnbHH, featH, TfeatH, do_longH, Tdo_longH)
CheckTradeRatio(gnbHL, featH, TfeatH, do_shortH, Tdo_shortH)
CheckTradeRatio(gnbLH, featL, TfeatL, do_longL, Tdo_longL)
CheckTradeRatio(gnbLL, featL, TfeatL, do_shortL, Tdo_shortL)



def CheckTradeRatio(m1,x1,x2,y1,y2):
    res_x = m1.predict(x1)
    res_y = m1.predict(x2)
    cnt1 = sum(np.where(res_x == y1, 1, 0)) / len(y1)
    cnt2 = sum(np.where(res_y == y2, 1, 0)) / len(y2)
    amnt1 = sum(np.where((res_x == y1) & (y1 == 1), 1, 0)) / sum(res_x)
    amnt2 = sum(np.where((res_y == y2) & (y2 == 1), 1, 0)) / sum(res_y)
    act1 = sum(y1) / len(y1)
    act2 = sum(y2) / len(y2)
    od1 = amnt1 / act1
    od2 = amnt2 / act2
    tr1 = sum(y1)
    tr2 = sum(y2)
    pt1 = sum(res_x)
    pt2 = sum(res_y)
    print("specificity train: %s, test: %s." % (cnt1, cnt2))
    print("Prop. good trades train: %s, test: %s." % (act1, act2))
    print("Pred. good trades train: %s, test: %s." % (amnt1, amnt2))
    print("Odds Improve train: %s, test: %s" % (od1, od2))
    print("Actual trades train: %s, test: %s" % (tr1, tr2))
    print("Predct trades train: %s, test: %s" % (pt1, pt2))
     

def BuildFeatures(sinds, twind):
    cls = ['ATR_SC', 'GBP_PROP', 'GBMAS', 'EUMAS', 'GBRSI', 'EURSI', 'EUATR', 'GBATR']
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
        features.iloc[i]['EUMAS'] = np.log(eucl / dfeu.loc[ind]['MAS'])
        features.iloc[i]['GBRSI'] = dfgb.loc[ind]['RSI']
        features.iloc[i]['EURSI'] = dfeu.loc[ind]['RSI']
        features.iloc[i]['EUATR'] = dfeu.loc[ind]['ATR']
        features.iloc[i]['GBATR'] = dfgb.loc[ind]['ATR']
    
    return [features, minmaxPrice]

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

def MultiTrig(ATRscH, ATRscL, twind):
    sH = SetTriggers(ATRscH, twind, 1)
    sL = SetTriggers(ATRscL, twind, -1)
    
    fH = BuildFeatures(sH, twind)
    fL = BuildFeatures(sL, twind)
    
    return [fH, fL, sH, sL]

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
    

    
    #return the maximisable mean of both NMSE
    return [HH, HL, LH, LL]

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