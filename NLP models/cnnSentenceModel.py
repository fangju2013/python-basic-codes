from __future__ import division
import numpy as np
from DeepLearning.BasicNeuron import baseNeuronLayer
from DeepLearning.CnnNeuron import  wideCovLayer,DCnnPooLayer,covFullLayer,covActLayer
#from NLP.statisticLanguageModel import statisLM
from sklearn.metrics import roc_auc_score
#-----------------------------------------------------------------------------------------------------------------------------------------------
#'''get corpus and get a dynamic word vec'''
#from nltk.corpus import brown
#corps = brown.sents(categories=None)
#corps = list(corps)
#slm = statisLM(corps,100)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#'''pick trained word vec'''
import cPickle
dirs = "C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\Python\\NLP\\savedObject\\brownCorpus\\"
slm = cPickle.load(open(dirs+"slm.pkl","rb"))
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''
hyper parameters
''' 
 
wordDim    = 100
outDim     = 1
outlayer   = baseNeuronLayer(100,outDim,actfunc='sigmoid')
actlayer   = covActLayer(actfunc='tanh')
fullayer   = covFullLayer(100,2,3,10)
dcpoolayer = DCnnPooLayer(3)
cnnlayer   = wideCovLayer(10,2,5,100)
learning   = 0.1
l2         = 0.0001
#-----------------------------------------------------------------------------------------------------------------------------------------------      
'''
training
'''       
pred = [] 
true = []
ers = 0
while 2>1:     
    for minibatch in np.random.choice(len(slm.codeCorps),1):  
        corp = slm.codeCorps[minibatch]
#    while 2>1:
        if len(corp)>5:     
            corp            = slm.codeCorps[minibatch]
            sentMatrix      = np.array([slm.wordvec[corp]])
            newSentMatrix   = cnnlayer.actNuerous(sentMatrix)
            newSentMatrix   = dcpoolayer.actNuerous(newSentMatrix,0)
            newSentMatrix   = fullayer.actNuerous(newSentMatrix)
            newSentMatrix   = actlayer.actNuerous(newSentMatrix)
            output          = outlayer.actNuerous(newSentMatrix)
            error           = output-1
            pred           += [np.abs(output[0][0])]
            true           += [1]
            ers            +=  np.abs(error[0][0]) 
            newError        = outlayer.updateError(error,learning,l2)
            newError        = actlayer.updateError(newError)
            newError        = fullayer.updateError(newError)
            newError        = dcpoolayer.updateError(newError)
            wordvecError    = cnnlayer.updateError(newError)
            fullayer.updateWeight(learning,l2)    
            cnnlayer.updateWeight(learning,l2)
#            slm.wordvec[corp]-=(wordvecError[0]*learning+slm.wordvec[corp]*l2)
            
            fakecorp        = slm.getFakeContext(corp)    
            sentMatrix      = np.array([slm.wordvec[fakecorp]])
            newSentMatrix   = cnnlayer.actNuerous(sentMatrix)
            newSentMatrix   = dcpoolayer.actNuerous(newSentMatrix,0)
            newSentMatrix   = fullayer.actNuerous(newSentMatrix)
            newSentMatrix   = actlayer.actNuerous(newSentMatrix)
            output          = outlayer.actNuerous(newSentMatrix)
            error           = output-0
            pred           += [np.abs(output[0][0])]
            true           += [-1]
            ers            +=  np.abs(error[0][0]) 
            newError        = outlayer.updateError(error,learning,l2)
            newError        = actlayer.updateError(newError)
            newError        = fullayer.updateError(newError)
            newError        = dcpoolayer.updateError(newError)
            wordvecError    = cnnlayer.updateError(newError)
            fullayer.updateWeight(learning,l2)    
            cnnlayer.updateWeight(learning,l2)   
#            slm.wordvec[fakecorp]-=(wordvecError[0]*learning+slm.wordvec[fakecorp]*l2)
            if len(pred)>=100:
                print roc_auc_score(np.array(true),np.array(pred)),ers/len(pred)
                pred = [] 
                true = []    
                ers = 0
#-----------------------------------------------------------------------------------------------------------------------------------------------      
'''
tesing
'''            
slm.getMostSimilarWord('she',types='distance')


 
