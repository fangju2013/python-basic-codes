# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:21:07 2016

@author: changlue.she
"""

from urllib  import urlopen
from bs4 import BeautifulSoup
from crawler.downloadPage import JDpage
'''get the first class categories'''
html  = urlopen('https://www.jd.com/')
bsObj = BeautifulSoup(html.read()) 
firstClassContents =  bsObj.find('ul',{'class':'JS_navCtn cate_menu'}).findAll('li',{'class':'cate_menu_item'})
firstClassList = [line.text for line in firstClassContents]
'''get the second,third and brand class categories'''
loopTime = 0
allClass = []
bsObj = BeautifulSoup(JDpage)
secondClassContents = bsObj.findAll('div',{'class':'cate_part_col1'})
for idx,secondClassContent in enumerate(secondClassContents):  
    '''first loop to get second class content and first class items'''
    firstclass = firstClassList[idx]  
    cateDetailContent = secondClassContent.find('div',{'class':'cate_detail'})
    cateDetailContents = cateDetailContent.findAll('dl')
    for cateDetailContent in cateDetailContents:
        '''second loop to get tird class content and second class items'''
        secondClass = cateDetailContent.find('dt',{'class':'cate_detail_tit'}).text[:-1]
        thirdClassContents = cateDetailContent.find('dd',{'class':'cate_detail_con'}).findAll('a') 
        for thirdClassContent in thirdClassContents:
            '''
            third loop to get brand content and third class items
            but, maybe there is not brand in a third class,so use try
            and except to handle those error
            '''
            loopTime+=1
            try:
                thirdClass = thirdClassContent.text
                url = 'https:'+thirdClassContent['href']
                html = urlopen(url)
                bsObj = BeautifulSoup(html)                
                brands = bsObj.find('ul',{'class':'J_valueList v-fixed'}).findAll('li')
                brands = [brand.text for brand in brands]
                for brand in brands:
                    allClass.append([firstclass,secondClass,thirdClass,brand])
            except:
                print 'there are no brands here',loopTime
#------------------------------------------------------------------------------
'''pick save and load the brands'''
#import cPickle
#dirs = "C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\Python\\NLP\\savedObject\\JD_TB\\"
#cPickle.dump(allClass,open(dirs+"jdBrands.pkl","wb")) 
#allClassLoad = cPickle.load(open(dirs+"jdBrands.pkl","rb"))
#for brands in allClassLoad:
#    print '->'.join(brands)
#------------------------------------------------------------------------------
'''save the brands into csv'''
#import pandas as pd
#BrandDF = pd.DataFrame(allClassLoad)
#BrandDF.to_csv('C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\MSXF\\feature construct\\doc\\brands.csv',encoding='gbk')
