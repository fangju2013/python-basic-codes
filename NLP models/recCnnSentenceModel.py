from __future__ import division
import numpy as np
from DeepLearning.BasicNeuron import baseNeuronLayer
from DeepLearning.CnnNeuron import recCnn
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
windowDim  = 5
actfunc    = 'tanh' 
outlayer   = baseNeuronLayer(100,1,actfunc='sigmoid')
#hiddenlayer= baseNeuronLayer(100,100,actfunc='tanh')
cnnlayer   = recCnn(wordDim,windowDim,actfunc)
learning   = 0.1
l2         = 0.001
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
        if len(corp)>5:     
            corp            = slm.codeCorps[minibatch]
            sentMatrix      = slm.wordvec[corp]
            newSentMatrix   = cnnlayer.actNuerous(sentMatrix)     
#            newSentMatrix   = hiddenlayer.actNuerous(newSentMatrix)  
            output          = outlayer.actNuerous(newSentMatrix)
            error           = output-1
            pred           += [np.abs(output[0][0])]
            true           += [1]
            ers            +=  np.abs(error[0][0]) 
            newError        = outlayer.updateError(error,learning,l2)   
#            newError        = hiddenlayer.updateError(newError,learning,l2)   
            wordvecError    = cnnlayer.updateError(newError) 
            cnnlayer.updateWeight(learning,l2)
            slm.wordvec[corp]-=(wordvecError[0]*learning+slm.wordvec[corp]*l2)
            
            fakecorp        = corp[::-1]   
            sentMatrix      = slm.wordvec[fakecorp] 
            newSentMatrix   = cnnlayer.actNuerous(sentMatrix)
#            newSentMatrix   = hiddenlayer.actNuerous(newSentMatrix) 
            output          = outlayer.actNuerous(newSentMatrix)
            error           = output-0
            pred           += [np.abs(output[0][0])]
            true           += [-1]
            ers            +=  np.abs(error[0][0]) 
            newError        = outlayer.updateError(error,learning,l2) 
#            newError        = hiddenlayer.updateError(newError,learning,l2)  
            wordvecError    = cnnlayer.updateError(newError)   
            cnnlayer.updateWeight(learning,l2)   
            slm.wordvec[fakecorp]-=(wordvecError[0]*learning+slm.wordvec[fakecorp]*l2)
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


 
