# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:38:56 2016

@author: changlue.she
"""
from __future__ import division
from DeepLearning.activeFunctions import actfuncs
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------------------------------
class baseNeuronLayer():
    def __init__(self,inputDim,outDim,actfunc='sigmoid'):
        self._activeFunc = actfuncs.getActFunc(actfunc)
        np.random.seed(123)
        self.W = np.random.uniform(0,0.1,size=(inputDim,outDim))
        self.b = np.random.uniform(0,0.1,size=(1,outDim))

    def actNuerous(self,inputs):
        self.inputs = inputs
        outputcore = self.inputs.dot(self.W)+self.b
        outputs,self.output_g = self._activeFunc(outputcore)
        return outputs

    def updateError(self,outputError,learning,l2=0):
        outputcoreError = outputError*self.output_g
        inputError = outputcoreError.dot(self.W.T)
        self.W -= (self.inputs.T.dot(outputcoreError)*learning+self.W*l2)
        self.b -= (np.sum(outputcoreError,axis=0)*learning+self.b*l2)       
        return inputError  
#-----------------------------------------------------------------------------------------------------------------------------------------------   
class BasicLayer():
    def __init__(self,inputDim,outDim,actfunc='sigmoid'):
        self._activeFunc = actfuncs.getActFunc(actfunc)
        np.random.seed(123)
        self.W = np.random.uniform(0,0.1,size=(inputDim,outDim))
        self.b = np.random.uniform(0,0.1,size=(1,outDim))
        self.deltaW = np.zeros(shape=self.W.shape)
        self.deltab = np.zeros(shape=self.b.shape) 
    def actNuerous(self,inputs):
        self.inputs = inputs
        outputcore = self.inputs.dot(self.W)+self.b
        outputs,self.output_g = self._activeFunc(outputcore)
        return outputs
        
    def updateError(self,outputError):
        outputcoreError = outputError*self.output_g
        inputError = outputcoreError.dot(self.W.T)
        self.deltaW += self.inputs.T.dot(outputcoreError)
        self.deltab += np.sum(outputcoreError,axis=0)
        return inputError 
    def updateWeight(self,learning,l2):
        self.W -= (self.deltaW*learning+self.W*l2)
        self.b -= (self.deltab*learning+self.b*l2)
        self.deltaW = np.zeros(shape=self.W.shape)
        self.deltab = np.zeros(shape=self.b.shape)
#-----------------------------------------------------------------------------------------------------------------------------------------------   
class softmaxLayer():
    def __init__(self):
        pass
    def actNuerous(self,inputs):
        expInputs = np.exp(inputs)
        expSum = np.sum(expInputs,axis=1).reshape(inputs.shape[0],1)
        self.softmaxOut = expInputs/expSum
        return self.softmaxOut
    def updateError(self,outputError):
        sumE = np.sum(self.softmaxOut*outputError,axis=1).reshape(outputError.shape[0],1)
        inputError = self.softmaxOut*(outputError-sumE)
        return inputError
