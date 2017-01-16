from __future__ import division
import numpy as np
from nltk.corpus import brown
from DeepLearning.BasicNeuron import BasicLayer
from DeepLearning.RnnNeuron import recCovlayer 
from NLP.statisticLanguageModel import statisLM
#-----------------------------------------------------------------------------------------------------------------------------------------------
#corps = brown.sents(categories=None)
#corps = list(corps)
slm = statisLM(corps,100)
#slm.initialWordHash(2)
#slm.getWordHash(10)
#onehotvec = np.eye(len(slm.word2code))
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''
hyper parameters
''' 
sematicDim =100
 
outs = len(slm.word2code )
 

rnnlayer = recCovlayer(sematicDim,sematicDim,maxSentLength=10,actfunc='tanh')
outlayer = BasicLayer(sematicDim,outs,actfunc = 'softmax')
#-----------------------------------------------------------------------------------------------------------------------------------------------      
'''
training
'''      
l2=0.0001
learning=0.05 
ers = 0
cot = 0 
 
while 2>1:     
    for minibatch in np.random.choice(len(slm.codeCorps),1):         
        corp = slm.codeCorps[minibatch]
        if len(corp)>5:       
            for widx in range(5,len(corp)):
                context    = corp[:widx]                
                yidx       = corp[widx]                
                sentMatrix = slm.wordvec[context]   
                y          = onehotvec[yidx]                
                sentMatrix = rnnlayer.actNuerous(sentMatrix)    
                output     = outlayer.actNuerous(sentMatrix)                
                outputError= output - y 
                ers       += np.abs(outputError[0][yidx])
                cot       += 1                
                inputError = outlayer.updateError(outputError)            
                inputError = rnnlayer.updateError(inputError)
                slm.saveWordVecChange(inputError,context,learning,l2)                
                if cot>100:
                    outlayer.updateWeight(learning,l2) 
                    rnnlayer.updateWeight(learning,l2) 
                    slm.changeWordVec()
                    print ers/cot
                    cot = 0
                    ers = 0                    
#-----------------------------------------------------------------------------------------------------------------------------------------------      
'''
tesing
'''          
def getSents(presents,ls=10):
    sents          = [slm.word2code[word] for word in presents]
    for i in range(ls):
        sentMatrix = slm.wordvec[sents]   
        sentMatrix = rnnlayer.actNuerous(sentMatrix)    
        output     = outlayer.actNuerous(sentMatrix)
        widx       = np.argmax(output[0])
        sents     += [widx]
    sents      = [slm.code2word[code] for code in sents]
    print ' '.join(sents)
#-----------------------------------------------------------------------------------------------------------------------------------------------      
presents = ['she','is','a','sad','girl','because']
getSents(presents,20)
slm.getMostSimilarWord('before',types='distance')
