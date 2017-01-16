# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:05:46 2016

@author: changlue.she
"""
from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
u'dsads'+str(1)+u'dsa'
class constructFrame():
    def __init__(self,docdir,tbns):
        self.connects = dict()
        self.paths = dict()
        self.tbns = list(set(tbns))
        self._getFrame(docdir)
    def _cleanDF(self,tb): 
        tb[u'字段英文名'] = tb[u'字段英文名'].apply(lambda x:x.upper()) 
        tb[u'字段中文名'][tb[u'字段中文名'].isnull()]='notRecord##' 
        tb[u'是否属于主键'] = tb[u'是否属于主键'].apply(lambda x:int(x=='Y'or x=='y')) 
        tb.loc[tb[u'字段英文名']=='ID',u'是否属于主键'] = 0
        return tb
    def _getFrame(self,docdir):
        self.wtb = pd.DataFrame(columns = [u'表名',u'字段中文名', u'是否属于主键', u'字段英文名', u'字段类型'])
        for tbn in self.tbns:
            print tbn
            tb  = pd.read_csv(docdir+tbn+'.csv',encoding='gbk')
            tb = tb[tb[u'字段英文名'].notnull()]
            tb = self._cleanDF(tb)
            tb[u'表名'] = tbn
            self.wtb = pd.concat([self.wtb, tb])
        self.etb = [self.tbns[0],self.tbns[1]]
        self._getConnect(self.tbns[0],self.tbns[1])
        for tb in self.tbns:
            if tb not in self.etb:
                self._constructConnect(tb)

    def findField(self,keyWords,ln=10):
        hittimes= []
        fields = np.array(self.wtb[u'字段中文名'])
        
        for field in fields:
            cot=0
            for kw in keyWords:
                if kw in field:
                    cot+=len(kw)
            hittimes.append(cot/len(field))
        hittimes = np.array(hittimes)
        print self.wtb.iloc[np.argsort(-hittimes)[:ln]]

    def _constructConnect(self,atb)  :
        for etb in self.etb[:]:
            self._getConnect(atb,etb)
        self.etb.append(atb)

    def _getConnect(self,tb1,tb2):
        wtb1 = self.wtb[self.wtb[u'表名']==tb1]
        wtb2 = self.wtb[self.wtb[u'表名']==tb2]

        tmp1 = wtb1[wtb1[u'是否属于主键']==1][u'字段英文名']
        tmp2 = wtb2[wtb2[u'是否属于主键']==1][u'字段英文名']
        if len(tmp1)>0 and tmp1.iloc[0] in list(wtb2[u'字段英文名']):
            self.connects.setdefault(tb1,{})
            self.connects[tb1][tb2] = tmp1.iloc[0]
        if  len(tmp2)>0 and tmp2.iloc[0] in list(wtb1[u'字段英文名']):
            self.connects.setdefault(tb2,{})
            self.connects[tb2][tb1] = tmp2.iloc[0]

    def _initpath(self):
        for tb in self.tbns:
            self.paths[tb]={tb:[tb,'stay',tb]}
    def findpath(self):
        self._initpath()
        for tb in self.connects.keys()[:]:            
            self._findpath(tb,tb,[tb])

    def _findpath(self,tb,tbp,path):
        if tbp in self.connects.keys():
            nowpath = path[:]
            for tbpair in self.connects[tbp].keys()[:]:
                if tbpair not in nowpath:
                    path = nowpath+[self.connects[tbp][tbpair],tbpair]
                    if tbpair not in self.paths[tb].keys():          
                        self.paths[tb][tbpair] = path
                    elif len(self.paths[tb][tbpair])>len(path):
                            self.paths[tb][tbpair] = path                    
                    self._findpath(tb,tbpair,path)

    def getpath(self,tbls):
        tbls = [tb.split('.')[0] for tb in tbls]
        endplace = []        
        for tb in tbls:
            if tb in self.paths.keys():
                endplace+=self.paths[tb].keys()        
        endplace = dict(Counter(endplace))
        endplace = max(endplace,key=endplace.get)        
        path = []
        for tb in tbls:
            if tb in self.paths.keys() and endplace in self.paths[tb].keys():
                if tb!=endplace:
                    path.append(self.paths[tb][endplace])
            else:
                print tb,"无法和其它表相关联"
        self._prettyPrint(path)         
        return path
    def _prettyPrint(self,paths):
        for path in paths:
            print "||".join(path)
    def path2Sql(self,paths,fileds):
        print '----------------------------------------------------------------------------'
        fileds = ','.join(fileds)
        print 'select',fileds,'from',paths[0][-1]
        idx = 1        
        for path in paths:
            ltb = paths[0][-1]
            for item in path[::-1][1:]:
                if np.mod(idx,2)==1:
                    key = item 
                    idx+=1
                elif np.mod(idx,2)==0:
                    tb = item 
                    print 'left join',tb
                    print 'on %s'%ltb+'.%s'%key+' = '+'%s'%tb+'.%s'%key
                    ltb = tb
                    idx+=1
        print '----------------------------------------------------------------------------'
                
            
