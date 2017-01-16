from __future__ import division
import numpy as np
from nltk.corpus import brown
from DeepLearning.activeFunctions import actfuncs
#-----------------------------------------------------------------------------------------------------------------------------------------------
rawcorpus = brown.sents(categories=None)
rawcorpus = list(rawcorpus)
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''hyperParameters'''
t = 0.0001
wordDim = 100
window = 5
NegSampNums = 10
'''clean corpus'''
corpus = []
for sentence in rawcorpus:
    corpus.append([word.lower() for word in sentence])
'''get word count'''
wordCount = {}
allWordCount = 0
for sentence in corpus:
    for word in sentence:
        wordCount.setdefault(word,0)
        wordCount[word]+=1
        allWordCount+=1
VocabNums = len(wordCount)
'''get word2code, code2word,codeCount and codeFreq'''
code2word = []
word2code = {}
codeCount = np.zeros(shape = VocabNums,dtype=int)
codeFreq  = np.zeros(shape = VocabNums)
code = 0
for word in wordCount:
    code2word.append(word)
    word2code[word] = code
    codeCount[code] = wordCount[word]
    codeFreq[code]  = wordCount[word]/allWordCount  
    code+=1       
'''get code corpus'''
codeCorpus = []
for sentence in corpus:
    codeCorpus.append([word2code[word] for word in sentence])
'''get wordvec and context vec'''
wordVec    = np.random.normal(size=(VocabNums,wordDim))
contextVec = np.random.normal(size=(VocabNums,wordDim))
'''get negtive samples sets'''
negtiveSampSets = []
for code in range(VocabNums):
    negtiveSampSets+=[code]*codeCount[code]
negtiveSampSets = np.array(negtiveSampSets)
'''subsampling word'''
def subSampCode(code):
    rat = t/codeFreq[code]
    if np.random.uniform(0,1)>1-np.power(rat,0.5)-rat:
        return True
    else:
        return False
'''training the word vec'''
ers = 0
cot = 0
Learning = 0.1
L2 = 0.0001
 
while 2>1:
    for sentence in codeCorpus:
        for wordloc,wordCode in enumerate(sentence):
            if subSampCode(wordCode):
                positivePairs             = sentence[max(0,wordloc-window):wordloc]+sentence[wordloc+1:wordloc+window]
                negtivePairs              = np.random.randint(VocabNums,size=len(positivePairs)*NegSampNums)
                negtivePairs              = [samp for samp in negtivePairs  if samp not in sentence]
                positivePairs             = [samp for samp in positivePairs if subSampCode(samp)]
                if len(negtivePairs)>3 and len(positivePairs)>0:
                    '''get wordvec'''
                    targVec                   = np.array([wordVec[wordCode]])    
                    postiveVec                = contextVec[positivePairs]
                    negtiveVec                = contextVec[negtivePairs]
                    postivePred,positiveGrad  = actfuncs.sigmoid(postiveVec.dot(targVec.T))
                    negtivePred,negtiveGrad   = actfuncs.sigmoid(negtiveVec.dot(targVec.T))
                    '''get errors'''
                    positiveErrors            = postivePred-1
                    negtiveErrors             = negtivePred-0
                    ers                      += np.mean(np.abs(positiveErrors))
                    cot                      += 1
                    positiveErrors           *= positiveGrad
                    negtiveErrors            *= negtiveGrad
                    if cot>10000:                
                        print ers/cot
                        ers = 0
                        cot = 0
                    postiveVecError           = positiveErrors.dot(targVec)
                    negtiveVecError           = negtiveErrors.dot(targVec)
                    targVecError              = postiveVec.T.dot(positiveErrors)+negtiveVec.T.dot(negtiveErrors)
                    wordVec[wordCode]        -= (targVecError[0]*Learning+wordVec[wordCode]*L2)
                    contextVec[positivePairs]-= (postiveVecError*Learning+contextVec[positivePairs]*L2)
                    contextVec[negtivePairs] -= (negtiveVecError*Learning+contextVec[negtivePairs]*L2)
                    
from scipy.spatial import distance as getDist
word = 'on'
code = word2code[word]
distList = []
for vec in  wordVec:
    distList.append(getDist.euclidean(wordVec[code],vec))
for code in np.argsort(distList)[1:10]:
    print code2word[code] 
