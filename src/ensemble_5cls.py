# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
from dataProcess import *
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier      
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# import pymrmr
import joblib
from sklearn.model_selection import StratifiedKFold
from prob import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from loading_Feature import *
from evaluate import *
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from intermedium import *


def sim_prob_vote(features, y, dataList,gbdt_index,xgb_index,modelPath,outpath):
    prob_list=base_train(features, y, dataList,gbdt_index,xgb_index,modelPath,'sim_prob_vote')
    prob=np.mean(np.swapaxes(np.array(prob_list),0,1),axis=1)
    pred = [int(i > 0.5) for i in prob] 
    a=fold_only(y,pred,prob,CV=10)
    res=cal_mean_var(a,['sim_prob_vote','10'])
    tofile(res,outpath+'sim_prob_vote'+'5cls'+'_sfs.txt',tohead=1)
    
    return res

def sim_pred_vote(features, y, dataList,gbdt_index,xgb_index,modelPath,outpath):
    prob_list=base_train(features, y, dataList,gbdt_index,xgb_index,modelPath,'sim_pred_vote')
    prob=np.swapaxes(np.array(prob_list),0,1)
    prob_list=[]
    pred_list=[]
    for i in range(prob.shape[0]):
        if sum(prob[i,:]>0.5) > sum(prob[i,:]<=0.5):
            pred_list.append(1)
            prob_list.append(max(prob[i,:]))
        else:
            pred_list.append(0)
            prob_list.append(min(prob[i,:]))            
    a=fold_only(y,pred_list,prob_list,CV=10)
    res=cal_mean_var(a,['sim_pred_vote','10'])
    tofile(res,outpath+'sim_pred_vote'+'5cls'+'_sfs.txt',tohead=1)
    
    return res

def linear_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain=0):    
    metric=[]
    namelist=[]
    prob_list=base_train(features, y, dataList,gbdt_index,xgb_index,mediumPath,'linear_vote',istrain)        
    feature=np.array(np.swapaxes(np.array(prob_list),0,1))
    modelList={'LR':'LogisticRegression'}
    if istrain:
        for key,modell in modelList.items():
            if os.path.exists(modelPath+str(key)+'_'+'linear_vote'+'_ensemble_sp2.model'):
                model=joblib.load(modelPath+str(key)+'_'+'linear_vote'+'_ensemble_sp2.model')
            else:
                
                if key=='SVC':
                    model=eval(modell)(random_state=1,probability=True)#,**q
                elif key=='KNeighborsClassifier'or 'MultinomialNB':
                    model=eval(modell)()#**q
                else:
                    model=eval(modell)(random_state=1)#,**q
                model=model.fit(feature,y)
                joblib.dump(model,modelPath+str(key)+'_'+'linear_vote'+'_ensemble_sp2.model')
            # lr_cls=LogisticRegression(random_state=1)        
            a,y_pred_prob_all=CV_res(model,feature, y,key,istrain=1)
            metric.append(a)
            namelist.append(['ave_vote',key])
        res=cal_mean_var(metric,namelist)
        tofile(res,outpath+'linear_vote'+'5cls'+'_sfs.txt',tohead=1)
        return res
    else:
        for key,modell in modelList.items():
            
            model=joblib.load(modelPath+str(key)+'_'+'linear_vote'+'_ensemble_sp2.model')
            a,y_pred_prob_all=CV_res(model,feature, y,key,istrain=0)
            print(*a)
        return y_pred_prob_all[:,1]

def ensemble_ways(features,y,dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain=0):
    
    #Simple Voting
    # a=sim_pred_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain)
    # a=sim_prob_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain)
    # bma_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain)
    a=linear_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain)
    # a = bma_vote(features, y, dataList,gbdt_index,xgb_index,mediumPath,modelPath,outpath,istrain)    
    # print(a)
    return a






    
    
    
    
    