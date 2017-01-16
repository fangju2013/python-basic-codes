# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 11:22:35 2016

@author: changlue.she
"""
from __future__ import division
import numpy as np
import cPickle
 
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''load and preprocess corps'''
corpus = cPickle.load(open("C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\pick1214sec.pkl","rb"))
wordCorps = []
for idx in corpus:
    for corp in corpus[idx]:
        if len(corp)>4:
            wordCorps.append(corp)
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''train and pick trained word vec'''
dirs = "C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\Python\\NLP\\savedObject\\CompCorpus\\"
slm = cPickle.load(open(dirs+"slm.pkl","rb"))
'''perform kmeans cluster without normalize'''
from sklearn.cluster import KMeans
TopicNums = 10
wordNums = slm.wordvec.shape[0]
kmeansFit = KMeans(n_clusters=TopicNums)
kmeansFit.fit(slm.wordvec)

'''perform hierachical cluster'''
import fastcluster 
result = fastcluster.linkage (X = slm.wordvec, method='single', metric='euclidean', preserve_input='False')
'''compute word depth'''
clustStruct = {}
for ridx in range(result.shape[0]):
    cidx = int(ridx+wordNums)
    clustStruct.setdefault(cidx,np.zeros(wordNums, dtype=np.int))
    for i in [0,1]:
        code = int(result[ridx][i])
        if code<wordNums:
            clustStruct[cidx][code]+=1
        else:
            clustStruct[cidx]+=(clustStruct[code]+ (clustStruct[code]!=0).astype('int'))
wordDepth = clustStruct[max(clustStruct.keys())]
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''compute the word degree within sentence co-occurance cross docs'''
sentWindow = 3
wordDegreeCollect = {}
for sentences in slm.codeCorps:
    for idx,word1 in enumerate(sentences):
        windowSentence = sentences[max(0,idx-sentWindow):idx+sentWindow]
        wordDegreeCollect.setdefault(word1,[])
        for word2 in windowSentence:
            wordDegreeCollect[word1].append(word2)
wordDegree = np.zeros(wordNums, dtype=np.int)
for word in wordDegreeCollect: 
     wordDegree[word] = len(set(wordDegreeCollect[word]))
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''compute the word score'''
'''set a and p values'''
wordMaxDepth = np.max(wordDepth)
wordMaxDegree = np.max(wordDegree)
for scale in np.arange(10):
    p=np.power(0.1,scale)
    print p,np.median(np.power(wordDepth/wordMaxDepth,p)),np.median(np.power(wordDegree/wordMaxDegree,p))
a = 1
p = 0.1
wordScore = np.zeros(wordNums)

for code in range(wordNums):
    wordScore[code] = np.power(wordDepth[code]/wordMaxDepth,a)*np.power(np.log(wordDegree[code])/np.log(wordMaxDegree),p)
'''compute the topic score'''
topicScore = np.zeros(TopicNums)
for topic in range(TopicNums):
    codes = np.where(kmeansFit.labels_==topic)[0] 
    topicScore[topic] = np.mean(wordScore[codes])
#-----------------------------------------------------------------------------------------------------------------------------------------------
'''print the results'''
Ktopics = 10
Kwords = 10
for topic in np.argsort(-topicScore)[:Ktopics]:
    codes = np.where(kmeansFit.labels_==topic)[0] 
    print 'topic',topic
    for code in codes[np.argsort(-wordScore[codes])[:Kwords]]:
        print slm.code2word[code]
