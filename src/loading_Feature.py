# -*- coding: utf-8 -*-
import pandas as pd
from loadfile import *
import numpy as np

def loading_feature(id_train,dataPath,haslabel=0):
    if haslabel:
        y=id_train['4']
    else:
        id_train['Gene name']=[0]*id_train.shape[0]
        id_train['4']=[0]*id_train.shape[0]
        y=id_train['4']
    id_train['chr']=id_train['chr'].astype(str)
    id_train['pos']=id_train['pos'].astype(str)
    id_train['unique'] = id_train['chr']+'_'+id_train['pos']+'_'+id_train['ref']+'_'+id_train['alt']      
    basic_feature=imfile(dataPath+'/features/all_features_filter.txt')  
    basic_feature=pd.merge(id_train,basic_feature,left_on=['unique'],right_on=[1],how='left').iloc[:,7].str.split(' ', expand=True)
    basic_feature = basic_feature.replace(to_replace='', value=np.nan)
    basic_feature = basic_feature.replace(to_replace='inf', value=np.nan)
    basic_feature=np.array(basic_feature,dtype='float32')
    
    return basic_feature,y   