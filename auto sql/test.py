# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:42:35 2016

@author: changlue.she
"""

from constructFrame import constructFrame
from readdata import docdir,tbns
cf = constructFrame(docdir,tbns)
cf.findpath()
######################################################################
#cf.findField([u'还款',u'金额'],10)
tbls = ['fdl_aprvadt_appl_main_chain.custorm_id','fdl_loanbor_term_chain.appl_no']
path = cf.getpath(tbls)
cf.path2Sql(path,tbls)


 
