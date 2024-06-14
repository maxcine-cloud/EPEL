# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
# import prob
# from arg import *
from loadfile import *
from dataProcess import *
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier      
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# import pymrmr
from evaluate import *
from prob import *

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os
modelList={'CatBoost':'CatBoostClassifier','XGB':'XGBClassifier','AdaBoost':'AdaBoostClassifier','RF':'RandomForestClassifier','GBDT':'GradientBoostingClassifier'}

def SFS(X_train,YtrainR,outpath,outname):
    
    Ytrain=YtrainR
    end=len(X_train.columns)
    flag=0
    for key,model in modelList.items():
        metric=[]
        for i in range(0,end):
            Xtrain=X_train.iloc[:,:i+1]
            if os.path.exists(outpath+'/model/'+str(i)+'_'+key+'mrmr2.model'):
                moduleDT=joblib.load(outpath+'/model/'+str(i)+'_'+key+'mrmr2.model')
            else:            
                moduleDT = eval(model)(random_state=1)  
                moduleDT.fit(Xtrain, Ytrain)    
                joblib.dump(moduleDT,outpath+'/model/'+str(i)+'_'+key+'mrmr2.model')
            [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]=CV_res(moduleDT,Xtrain,Ytrain,key,istrain=1)#
            metric.append([i,key,outname,np.round(Pre,3),np.round(recall,3),np.round(specificity,3),np.round(BACC,3),np.round(F1_Score,3),np.round(MCC,3),np.round(ACC,3),np.round(roc_auc,3),np.round(PRC,3)])
        # np.savetxt(outpath+'\\res\\'+key+outname+'2_'+'5cls'+'_sfs.txt',np.array(metric),delimiter='\t',fmt='%s')
    return flag,Xtrain.columns.tolist()

