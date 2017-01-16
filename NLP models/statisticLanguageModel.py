    # -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:16:04 2016

@author: changlue.she
"""
from __future__ import division
import numpy as np
from scipy.spatial import distance as getDist
 
 
 
class statisLM():
    def __init__(self,corps,wordDim ):
        self.corps = corps 
         
        self._drop_stopwls()
        self._getWordDict()
        self._Word2Code()
        np.random.seed(123)
        self.wordvec = np.random.uniform(0,1,size = (len(self.wordDict),wordDim)) 
        self.tmpWordVec = self.wordvec.copy()
        
    def _drop_stopwl(self,corp,stops=['.','!','?',';',',']):
        corp = [w.lower() for w in corp]
        while len(corp)>0 and corp[-1] in stops:
            corp = corp[:-1]
        return corp

    def _drop_stopwls(self):
        self.corps[:] = [self._drop_stopwl(corp) for corp in self.corps]

    def _getWordDict(self):
         self.wordDict = {}
         self.allWordCount = 0
         for corp in self.corps:
             for w in corp:
                 self.wordDict.setdefault(w,0)
                 self.wordDict[w]+=1
                 self.allWordCount+=1
 
           
    def _Word2Code(self):
        self.vocabNums = len(self.wordDict.keys())   
        self.word2code = {}
        self.code2word = []
        self.codeCorps = []
        self.codeFreq  = np.zeros(shape=self.vocabNums)         
        for idx,word in enumerate(self.wordDict.keys()):
            self.word2code[word] = idx
            self.code2word.append(word)
            self.codeFreq[idx] = self.wordDict[word]/self.allWordCount      
        self.codeCorps = [[self.word2code[w] for w in corp] for corp in self.corps]
    def subSampCode(self,code):
        rat = 0.00001/self.codeFreq[code]
        if np.random.uniform(0,1)>1-np.power(rat,0.5)-rat:
            return True
        else:
            return False     
    def getFakeContext(self,realcontext):       
         return [w if np.random.normal(1)<1 else self._getFakeWord(w,realcontext)for w in realcontext]
 
    def _getFakeWord(self,w,context):
        while 2>1:
            fw = np.random.choice(context,1)[0]
            if fw!=w:
                return fw 
    def saveWordVecChange(self,error,context,learning,l2):
        for row in range(len(error)):
            self.tmpWordVec[context[row]] -= error[row]*learning
        self.tmpWordVec[np.unique(np.array([context]))] -= self.tmpWordVec[np.unique(np.array([context]))]*l2
    def changeWordVec(self):
        self.wordvec = self.tmpWordVec.copy()
    def updateWordVec(self,error,context,learning,l2):
        for row in range(len(error)):
            for col in range(len(error[row])):
                self.wordvec[context[col][row]] -=error[row][col]*learning
        self.wordvec[np.unique(np.array([context]).flatten())] -= self.wordvec[np.unique(np.array([context]).flatten())]*l2 
    def printMostSimilarPair(self,mostSimilarPair):
        avgSimilar = np.mean(self.distArray,axis=0)
        for codei in np.argsort(avgSimilar)[:mostSimilarPair]:
            codes = np.argsort(self.distArray[codei])[1:mostSimilarPair]
            words = [self.code2word[code] for code in codes]
            print self.code2word[codei]+' : '+'|'.join(words)
         
    def getMostSimilarWord(self,word='',lis=10,types='distance'):  
        if word=='':
             self.distArray = np.zeros(shape=(self.vocabNums,self.vocabNums))
             for codei in range(self.vocabNums):
                 for codej in range(self.vocabNums):
                     if types=='distance':
                         self.distArray[codei][codej] = getDist.euclidean(self.wordvec[codei],self.wordvec[codej])
                     else:
                         self.distArray[codei][codej] = getDist.cosine(self.wordvec[codei],self.wordvec[codej])
        else:
            code = self.word2code[word]
            vec1 = self.wordvec[code]
            distVec = []
            for vec2 in self.wordvec:                                
                if types=='distance':
                    distVec.append(getDist.euclidean(vec1,vec2))
                elif types=='cosSimilar':
                    distVec.append(getDist.cosine(vec1,vec2))     
            for code in np.argsort(distVec)[1:lis]:
                print self.code2word[code] 
            
 
    def initialWordHash(self,ngram = 5):        
        self.word2ngchar = {}  
        self.ngcharDict = {}
        for word in self.wordDict:
            self.word2ngchar.setdefault(word,[])
            if len(word)>ngram-3:                    
                addword = '&'+word+'*'
                for idx in range(len(addword)-ngram+1):
                    
                    self.word2ngchar[word].append(addword[idx:idx+ngram])
                    self.ngcharDict.setdefault(addword[idx:idx+ngram],0)
                    self.ngcharDict[addword[idx:idx+ngram]]+=1
        
    def getWordHash(self,freq=200): 
        while freq!='n':    
            freq = int(freq)
            self.ngchar2code = {}       
            idx=0   
            for gr in self.ngcharDict:
                self.ngchar2code.setdefault(gr,-1)
                if self.ngcharDict[gr]>freq:
                    self.ngchar2code[gr]=idx
                    idx+=1      
            print idx
            freq = raw_input('continue to find a good word hash dimemsion?')
        self.wordhash = np.zeros(shape=(len(self.wordDict),idx+1))
        for word in self.word2code:
            for ngchar in self.word2ngchar[word]:             
                self.wordhash[self.word2code[word]][self.ngchar2code[ngchar]] = 1
