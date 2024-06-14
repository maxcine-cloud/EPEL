
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

modelList={'CatBoost':'CatBoostClassifier','XGB':'XGBClassifier','AdaBoost':'AdaBoostClassifier','RF':'RandomForestClassifier','GBDT':'GradientBoostingClassifier'}

def base_train(features, y, dataList,gbdt_index,xgb_index,modelPath,vote_ways,istrain=0):
    prob_list=[]

    for key,model in modelList.items():
        if istrain:
            if os.path.exists(modelPath+str(key)+'_'+vote_ways+'_ensemble.model'):
                modell=joblib.load(modelPath+str(key)+'_'+vote_ways+'_ensemble.model')
            else:    
                modell=eval(model)(random_state=1) 
                modell.fit(features[:,dataList[key]], y)
                joblib.dump(modell,modelPath+str(key)+'_'+vote_ways+'_ensemble.model')
            a,y_pred_prob_all=CV_res(modell,features[:,dataList[key]], y,model,istrain=1)
            prob_list.append(y_pred_prob_all[:,1])
        else:
            modell=joblib.load(modelPath+str(key)+'_'+vote_ways+'_ensemble.model')
            _,predDT_proba_t=CV_res(modell,features[:,dataList[key]], y,key,istrain=0)
            prob_list.append(predDT_proba_t[:,1])        
    return  prob_list
       