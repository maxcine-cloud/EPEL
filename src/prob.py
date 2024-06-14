# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import evaluate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression



def CV_res(moduleDT,X,y,i,istrain=False):
    tem=[i]
    predDT = moduleDT.predict(X)
    predDT_proba=moduleDT.predict_proba(X)
    if istrain:
        a,predDT_proba_cv=evaluate.fold2(moduleDT,X,y,CV=10)
        return a,predDT_proba_cv
    else:
        [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]=evaluate.metr1_2(y,predDT,predDT_proba[:,1])        
        tem=[np.round(Pre,3),np.round(recall,3),np.round(specificity,3),np.round(BACC,3),np.round(F1_Score,3),np.round(MCC,3),np.round(ACC,3),np.round(roc_auc,3),np.round(PRC,3)]
        return tem,predDT_proba