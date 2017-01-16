# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:39:06 2016

@author: changlue.she
"""
from __future__ import division
from activeFunctions import actfuncs
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------------------------------
class recCovlayer():
    def __init__(self,wordDim,sematicDim,maxSentLength=10,actfunc='tanh'):
        self.sematicDim    = sematicDim
        self.wordDim       = wordDim
        self.maxSentLength = maxSentLength
        self._actfunc      = actfuncs.getActFunc(actfunc)   
        np.random.seed(123)
        self.W      = np.random.uniform(-0.001,0.001,size=(maxSentLength,wordDim,sematicDim)) 
        self.b      = np.random.uniform(-0.1,0.1,size=(1,sematicDim))
        self.deltaW = np.zeros(shape=self.W.shape)
        self.deltab = np.zeros(shape=self.b.shape)
    def actNuerous(self,wordSematic):
        self.wordSematic = wordSematic
        hiddenSematic = np.zeros(shape=(1,self.sematicDim))
        for widx in range(min(self.maxSentLength,len(wordSematic))):
            hiddenSematic += np.array([wordSematic[-widx-1]]).dot(self.W[widx]) 
        sematicOutput,self.sematicOutput_g = self._actfunc(hiddenSematic+self.b)
        return sematicOutput
    def updateError(self,sematicOutputError):    
        wordError = np.zeros(shape=self.wordSematic.shape)
        sematicOutputError *= self.sematicOutput_g
        self.deltab += np.sum(sematicOutputError,axis=0)
        for widx in range(min(self.maxSentLength,len(self.wordSematic))):
            self.deltaW[widx]  += np.array([self.wordSematic[-widx-1]]).T.dot(sematicOutputError)
            wordError[-widx-1] += sematicOutputError.dot(self.W[widx].T)[0]
        return  wordError
        
    def updateWeight(self,learning,l2):
        self.W -= (self.deltaW*learning+self.W*l2)
        self.b -= (self.deltab*learning+self.b*l2)
        self.deltaW = np.zeros(shape=self.W.shape)
        self.deltab = np.zeros(shape=self.b.shape)

#-----------------------------------------------------------------------------------------------------------------------------------------------
class simpRnn():
    def __init__(self,sematicDim,window=5,actfunc='tanh'):  
        self.window   = window
        self._actfunc = actfuncs.getActFunc(actfunc)  
        np.random.seed(123)
        self.c = np.random.uniform(-0.1,0.1,size=(1,sematicDim))
        self.U = np.random.uniform(-0.1,0.1,size=(sematicDim,sematicDim))
    def actNuerous(self,wordSematic):
        self.wordSematic = wordSematic        
        self.hiddenSematic = np.zeros(shape=(wordSematic.shape[0]+1,wordSematic.shape[1]))
        self.hiddenSematic_g = np.zeros(shape=wordSematic.shape)
        for idx in range(wordSematic.shape[0]):
            self.hiddenSematic[idx+1],self.hiddenSematic_g[idx]= self._actfunc(self.hiddenSematic[idx].dot(self.U)+wordSematic[idx]+self.c)   
        return self.hiddenSematic[1:]
    def updateError(self,sematicOutputError,learning,l2):    
        wordError = np.zeros(shape=self.wordSematic.shape)
        deltac = np.zeros(shape=self.c.shape)
        deltaU = np.zeros(shape=self.U.shape)
        for outidx in range(self.window,sematicOutputError.shape[0])[::-1]:
            hiddenError = sematicOutputError[outidx] 
            for widx in range(self.window+1):
                hiddenCoreError = hiddenError*self.hiddenSematic_g[outidx-widx]
                hiddenError = hiddenCoreError.dot(self.U.T)
                wordError[outidx-widx]+=hiddenCoreError
                deltac[0] +=hiddenCoreError
                deltaU    +=self.hiddenSematic[outidx-widx].T.dot(hiddenCoreError)
        self.c-=(deltac*learning+self.c*l2) 
        self.U-=(deltaU*learning+self.U*l2) 
        return  wordError
   

