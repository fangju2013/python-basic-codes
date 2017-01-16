# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:16:03 2016

@author: changlue.she
"""
import numpy as np
from DeepLearning.CnnNeuron import recCnn  
###############################################################################
wordDim   = 100
windowDim = 5
actfunc   = 'tanh' 
dc1 = recCnn(wordDim,windowDim,actfunc)
dc2 = recCnn(wordDim,windowDim,actfunc)
np.random.seed(123)   
for n in range(2,windowDim+1):
    dc1.W[n] = np.random.uniform(-1,1,size=(n,wordDim))
    dc1.b[n] = np.random.uniform(-1,1,size=(wordDim))  
X_real = np.random.normal(-1,1,size=(20,100))
X_train = np.random.uniform(0,1,size=(20,100)) 
###############################################################################
y_real = dc1.actNuerous(X_real) 
############################################################################### 
l2 =  0.0001
learning = 0.01
###############################################################################
for minibatch in np.random.choice(100,10000):
#    ers = np.mean(np.abs(dc2.actNuerous(X_real)-y_real))
    dc2.W = dc1.W.copy()
    dc2.b = dc1.b.copy()
#    dc2.U = dc1.U.copy()
#    dc2.c = dc1.c.copy()
    y_pred = dc2.actNuerous(X_train)
    outError = y_pred - y_real 
    
    inputError  = dc2.updateError(outError)   
#    dc2.updateWeight(learning,l2)
    X_train -= (inputError*learning+X_train*l2)
    evals = np.mean(np.abs(outError))                                
    print evals 
 
 
        
