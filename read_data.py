# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:19:49 2019

1-Min ohlc data downloaded from #data downloaded 
from https://forextester.com/data/datasources

@author: Oliver
"""
 
import pandas as pd

eurs = pd.read_csv("EURUSD/EURUSD.csv")
gbps = pd.read_csv("EURUSD/GBPUSD.csv")

#There are various missing rows in both csvs, setting index to unique DT
#Then inner join on index
eurs.index = eurs['<DTYYYYMMDD>'].astype(str) + "_" + eurs['<TIME>'].astype(str)
gbps.index = gbps['<DTYYYYMMDD>'].astype(str) + "_" + gbps['<TIME>'].astype(str)

#create an inner join index without duplicating all the data (my pc is struggling with RAM)
bth = pd.concat([eurs['<TICKER>'],gbps['<TICKER>']], join='inner', axis=1,copy=False)

eurs = eurs.loc[bth.index]
gbps = gbps.loc[bth.index]

bth = None

#this function collapses time-frames
def collapseMinutes(dat, interval):
    ln = dat.shape[0]
    opens = dat[0:(ln-interval):interval].xs('<OPEN>', axis=1)
    outmin = pd.DataFrame()
    
    outmin['CL'] = dat[(interval-1)::interval].xs('<CLOSE>', axis=1)
    outmin['HI'] = dat['<HIGH>'].rolling(window=interval).max()[(interval-1)::interval]
    outmin['LO'] = dat['<LOW>'].rolling(window=interval).min()[(interval-1)::interval]
    outmin['VL'] = dat['<VOL>'].rolling(window=interval).sum()[(interval-1)::interval]
    opens.index = pd.RangeIndex(len(opens))
    outmin.index = pd.RangeIndex(len(opens))
    outmin['OP'] = opens
    
    return outmin

#These are the time-steps we can try to trade at (multiples of minutes)
#intervals = 15min, 1hr, 4hr, 1D
intervals = [15,60,240,1440]

#These are the lists which will contain the four time-step matrices
eu_data = []
gb_data= []

#This loop generates the collapsed timeframe data
for i in range(4):
    eu_data.append(collapseMinutes(eurs, intervals[i]))
    gb_data.append(collapseMinutes(gbps, intervals[i]))
    
#set index to numerical ascending for easy index->interval conversions
eurs.index = pd.RangeIndex(eurs.shape[0])
gbps.index = pd.RangeIndex(gbps.shape[0])

gbps.columns = ['<xTICKER>', '<xDTYYYYMMDD>', '<xTIME>', '<xOPEN>', '<xHIGH>', '<xLOW>','<xCLOSE>', '<xVOL>']

#bth2 = eurs.join(gbps)
#bth2.to_csv("all_currencies.txt")
