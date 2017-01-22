# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:59:47 2017

@author: changlue.she
"""
import jieba.posseg as pseg
import jieba
import pandas as pd
 
jieba.load_userdict(dir)  
#---------------------------------------------------------------------------------------------------
rawdata = pd.read_csv(dir,
                      header=0,encoding='gbk',usecols=[0,1,2,3,4,5,6 ])
rawdata = rawdata.loc[rawdata[u'消息目标'] == u'机器人',:]    
rawdata = rawdata.sort([u'会话ID'])
rawdata = rawdata.drop_duplicates()
#---------------------------------------------------------------------------------------------------
corpus = []
sentences = []
userIDs = list(rawdata[u'会话ID'])
conversations = list(rawdata[u'消息内容'])
for sentence in conversations:
    sentence = sentence.replace('\t','')
    words = [pair.word for pair in pseg.lcut(sentence) if pair.flag in ['n','ns','vs','nv']]       
    if len(words)>2:
         corpus.append(words)
         sentences.append(sentence)
 
