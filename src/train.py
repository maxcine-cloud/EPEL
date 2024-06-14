# -*- coding: utf-8 -*-

import numpy as np
import intermedium
from arg import *
from dataProcess import *
from loadfile import *
from dataProcess import *
from loading_Feature import *
from ensemble_5cls import *
import featureSelect as sfs

def train1():
    args=get_args()
    data=imfile(args.dataPath+'/id_train_close'+args.r+'.txt',hashead=1)
    features,y=loading_feature(data,args.dataPath,haslabel=1)
    features_del=cal_var(features,args.processingmodelPath+'/allFeatures_'+args.dimType+args.r,istrain=1)
    features=dataProcessing(features_del,args.processingmodelPath+'/allFeatures_'+args.dimType+args.r,istrain=1)   
    # sfs.clf_sfs(features,y,args.interdataPath)
    gbdt_index=[17,31,19,14,62,3,40,28,1,26,60,8]
    xgb_index=[17,135,19,3,14,93,62,100,86,96,117,40,60,106,34,56,11,67,31,8,109,44,42,71,91,89,47,48,2,33,28,15,32,74,20,45,5,41,4,66,130,24,9,21,39,12,26]
    dataList={'CatBoost':gbdt_index,'XGB':xgb_index,'AdaBoost':xgb_index[:4],'RF':xgb_index,'GBDT':gbdt_index}
    ensemble_ways(features,y,dataList,gbdt_index,xgb_index,args.intermodelPath+'/',args.clsmodelPath+'/',args.clsdataPath+'/',istrain=1)
    