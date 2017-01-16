# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:30:26 2016

@author: changlue.she
"""

from wordcloud import WordCloud
from load_data import QAdf
from PIL import Image
import numpy as np
picdir = 'C:\\Users\\Administrator.NBJXUEJUN-LI\\Desktop\\project\\MSXF\\intel_cust\\statistics\\pics\\'
text = ''
for i,cate in enumerate(QAdf[u'所属分类']):
    text += (' '+cate.split('/')[-1])*QAdf[u'命中次数'][i]
text = text.split(' ')
text = ' '.join(text)

alice_mask = np.array(Image.open(picdir+'msxf2.jpg'))
 
wordcloud = WordCloud(font_path='C:/Users/Administrator.NBJXUEJUN-LI/Desktop/project/intel_cust/pycode/simhei.ttf', 
                          background_color="white",  
                          mask=alice_mask).generate(text)
# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.imshow(wordcloud)
plt.axis("off")
