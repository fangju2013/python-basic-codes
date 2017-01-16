# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 15:25:22 2016

@author: changlue.she
"""
from __future__ import division
import numpy as np
from nltk.corpus import brown
from DeepLearning.BasicNeuron import baseNeuronLayer
from NLP.statisticLanguageModel import statisLM
#-----------------------------------------------------------------------------------------------------------------------------------------------
corps = brown.sents(categories=None)
corps = list(corps)
slm = statisLM(corps,100)
#-----------------------------------------------------------------------------------------------------------------------------------------------
maxNgram = 5 
sentWordhash = {}
for ngram in range(2,maxNgram): 
    for sent in slm.corps:
        if len(sent)-ngram>0:
            sent ='#'.join(sent)
            sent = '&'+sent+'*'
            for idx in range(len(sent)-ngram+1):
                sentWordhash.setdefault(sent[idx:idx+ngram],0)
                sentWordhash[sent[idx:idx+ngram]]+=1
#-----------------------------------------------------------------------------------------------------------------------------------------------
nchar2code = {}
idx = 0
for nchar in sentWordhash:
    if sentWordhash[nchar]>10:
        nchar2code[nchar] = idx
        idx+=1
#----------------------------------------------------------------------------------------------------------------------------------------------- 
hashCorps = []
for sent in slm.corps:
    if len(sent)>4:
        hashcorp = []
        sent ='#'.join(sent)
        sent = '&'+sent+'*'
        for ngram in range(2,maxNgram):
            for idx in range(len(sent)-ngram+1):
                if sent[idx:idx+ngram] in nchar2code:
                    hashcorp.append(nchar2code[sent[idx:idx+ngram]])
        hashCorps.append(hashcorp)
fakehashCorps = []
for sent in slm.corps:
    if len(sent)>4:
        sent = slm.getFakeContext(sent)
        hashcorp = []
        sent ='#'.join(sent)
        sent = '&'+sent+'*'
        for ngram in range(2,maxNgram):
            for idx in range(len(sent)-ngram+1):
                if sent[idx:idx+ngram] in nchar2code:
                    hashcorp.append(nchar2code[sent[idx:idx+ngram]])
        fakehashCorps.append(hashcorp)

#----------------------------------------------------------------------------------------------------------------------------------------------- 
from sklearn.metrics import roc_auc_score
hashVec  = np.random.uniform(0,1,size = (len(nchar2code),100)) 
outlayer = baseNeuronLayer(100,1,actfunc = 'sigmoid')
hiddenlayer = baseNeuronLayer(100,100,actfunc = 'tanh')
learning = 0.05
l2       = 0.00001

while 2>1:
    for senidx in range(0,len(hashCorps)):                       
        tmpx    = np.array([np.sum(hashVec[hashCorps[senidx]],axis=0),np.sum(hashVec[fakehashCorps2[senidx]],axis=0)])  
        tmpy    = np.array([[1],[0]]) 
        tmpindx = [hashCorps[senidx],fakehashCorps2[senidx]]
        if senidx == 0:
            y        = tmpy
            hashProj = tmpx
            hashidx  = tmpindx
        elif senidx%1000==0:
            hidden   = hiddenlayer.actNuerous(hashProj)
            out      = outlayer.actNuerous(hidden)
            error    = out - y
            y[y==0] = -1
#            print roc_auc_score(y.T[0],out.T[0])
            hiddError= outlayer.updateError(error,learning,l2)
            hashError= hiddenlayer.updateError(hiddError,learning,l2)
            print np.mean(np.abs(error))
            for hashid in range(len(hashError)):
                hashVec[hashidx[hashid]] -= (hashError[hashid]*learning+hashVec[hashidx[hashid]]*l2) 
            y        = tmpy
            hashProj = tmpx
            hashidx  = tmpindx
        else:
            hashProj = np.append(hashProj,tmpx,axis=0)
            y        = np.append(y,tmpy,axis=0)
            hashidx += tmpindx
         
        
        
