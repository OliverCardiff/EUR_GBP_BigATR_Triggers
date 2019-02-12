# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 09:20:06 2019

@author: Oliver
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

"""
res[1].best_estimator_
res[0].best_estimator_
res2[1].best_estimator_
res2[0].best_estimator_

res[1].best_estimator_.feature_importances_
res[0].best_estimator_.feature_importances_

np.sqrt(-res[0].best_score_)
np.sqrt(-res[1].best_score_)

"""
res2 = pkl.load(open('picks/1D_models.pickle', "rb"))

D_perf = pkl.load(open("picks/1D_perfList.pickle", "rb"))

len(D_perf)

plt.plot(D_perf[1])

cls = ['ATR_SC', 'GBP_PROP', 'GBMAS', 'GBMAF', 'EUMAS', 'EUMAF', 'GBRSI', 'EURSI', 'EUATR', 'GBATR']

plotImps(res[0].best_estimator_.feature_importances_, cls, "1H_MinPrice")
plotImps(res[1].best_estimator_.feature_importances_, cls, "1H_MaxPrice")

plotImps(res[0].best_estimator_.feature_importances_, cls, "4H_MinPrice")
plotImps(res[1].best_estimator_.feature_importances_, cls, "4H_MaxPrice")
plotImps(res2[0].best_estimator_.feature_importances_, cls, "1D_MinPrice")
plotImps(res2[1].best_estimator_.feature_importances_, cls, "1D_MaxPrice")

plotTstTrain(port.perfList, port.days, "1D_Train_Test")
plotTstTrain(port.perfList, port.days, "4H_Train_Test")

plotTstTrain(port.perfList, port.days, "1H_Train_Test")

def plotTstTrain(perfs, days, headr):
    fig = plt.figure()
    ptrain = int(len(days) * 0.39)
    
    ax = fig.add_subplot(111)
    ax.set_title(headr)
    ax.set_xlabel('Days (since Jan/01)')
    ax.set_ylabel('Portfolio Value ($)')
    
    for i in np.arange(0, len(perfs)):
        plt.plot(perfs[i][:ptrain])
    plt.plot(np.arange(ptrain,len(days)), days[ptrain:])
    plt.axvline(ptrain, color='r')
    
    plt.savefig(headr + '.png', dpi=150)

def plotImps(imp, cls, headr):
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.set_title(headr)
    ax.set_xlim([0.0,0.36])
    ax.set_xlabel('Importance (RFR)')
    ax.set_ylabel('Features')
    pd.Series(imp, cls).plot(kind='barh')
    
    plt.savefig(headr + '.png', dpi=150)
    
    
