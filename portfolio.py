# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 09:16:53 2019

This script defines a basic Portfolio class which one can use to track the
returns of the trading system. It has an active array (days) which is reset
and a memory (perfList) which stores all past performances on reset so
they can be accessed later (i.e. this allows the storage of all performance
series created during the parameter optimisation process)

@author: Oliver
"""
import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, initial, size):
        self.interval = 1440
        self.perfList = []
        self.initial = float(initial)
        self.days = np.asarray([self.initial] * size)
        self.size = size
        self.lastdeltaday = 0
        
    #Updates all future balances as trades come in
    def AddDelta(self, ind, d, interv):
        scfac = (self.interval/interv)
        adjust = int(ind/scfac)
        if(adjust >= len(self.days)):
            adjust = len(self.days) - 1
            
        self.RollForward(adjust, self.lastdeltaday)
        self.days[adjust] += d
        self.lastdeltaday = adjust
    
    #keeps no-trade days up to date
    def RollForward(self, ad, lst):
        if(lst != ad):
            self.days[(lst + 1):(ad+1)] = self.days[lst]
    #Returns the balance for the current day
    def GetCurrent(self, ind, interv):
        scfac = (self.interval/interv)
        adjust = int(ind/scfac)
        if(adjust >= len(self.days)):
            adjust = len(self.days) - 1
        return self.days[adjust]
    
    #Calc final % return centered on zero
    def GetFinalReturn(self):
        self.RollForward(len(self.days)-1, self.lastdeltaday)
        return ((self.days[-1]/self.days[0]) - 1) * 100
    
    #reset portfolio to the initial value, save the last run
    def ResetIt(self):
        self.perfList.append(np.copy(self.days))
        self.days = np.asarray([self.initial] * self.size)
        
    #empty the stored performances
    def Empty(self):
        self.perfList = []
        
class ParaSet:
    def __init__(self, names):
        self.D = []
        for i in range(len(names)):
            self.D.append(0.0)
    def Clear(self):
        for i in self.D:
            self.D[i] = 0
    def List(self):
        return pd.Series(self.D)
    def __copy__(self):
        return ParaSet(self.Qprof, self.Qloss, self.RR)
    def Dup(self):
        return self.__copy__()

class ParaMemory:
    def __init__(self, names, days):
        self.names = names
        self.PS = ParaSet(names)
        self.Plist = []
        self.folios = []
        self.PF = Portfolio(10000, days)
    def NewRun(self):
        self.Plist.append(self.PS.List())
        self.PS = ParaSet(self.names)
        self.folios.append(self.PF)
        self.PF = Portfolio(10000, self.PF.size)
    def GetDF(self):
        df = pd.concat(self.Plist)
        df.columns = self.names
        return df
    def Empty(self):
        self.PS = ParaSet()
        self.Plist = [] 
        
        
        
        
        
        
        
        
        
        
        
        
        
        