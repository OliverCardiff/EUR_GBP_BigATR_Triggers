# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:40:48 2019

This script colmun-joins the multi-timeframe historical data with some
standard technical indicators from TA-lib (http://ta-lib.org)

@author: Oliver
"""
import talib as tlb

#Indicator variables
atr_scale = 20
MA_slow = 150
MA_fast = 40
rsi_period = 14

##Append all indicators used to all timeframes
def appendRSI(xlist, per):
    for i in range(len(xlist)):
        t1 = xlist[i].as_matrix(['CL']).flatten()
        xlist[i]['RSI'] = tlb.RSI(t1,per)
        
def appendMASlow(xlist, slw):
    for i in range(len(xlist)):
        t1 = xlist[i].as_matrix(['CL']).flatten()
        xlist[i]['MAS'] = tlb.SMA(t1,slw)
        
def appendMAFast(xlist, fst):
    for i in range(len(xlist)):
        t1 = xlist[i].as_matrix(['CL']).flatten()
        xlist[i]['MAF'] = tlb.SMA(t1,fst)

def appendATR(xlist, scl):
    for i in range(len(xlist)):
        t1 = xlist[i].as_matrix(['CL']).flatten()
        t2 = xlist[i].as_matrix(['LO']).flatten()
        t3 = xlist[i].as_matrix(['HI']).flatten()
        xlist[i]['ATR'] = tlb.ATR(t3,t2,t1,scl)
        
        
appendRSI(eu_data, rsi_period)
appendRSI(gb_data, rsi_period)
appendMASlow(eu_data, MA_slow)
appendMASlow(gb_data, MA_slow)
appendMAFast(eu_data, MA_fast)
appendMAFast(gb_data, MA_fast)
appendATR(eu_data, atr_scale)
appendATR(gb_data, atr_scale)

print(eu_data[1].tail(4))
