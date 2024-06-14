#!/usr/bin/env python
# encoding:utf-8

import pandas as pd
import numpy as np
import numpy as np
import os
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def dataProcessing(data,i,istrain=0):
    """
    i:path
    data
    """
    if istrain==1:
        if os.path.exists(i+'.dataProcessing'):
            clf=joblib.load(i+'.dataProcessing')
            clf.transform(data)
        else:
            clf = Pipeline([('missing_values',SimpleImputer(missing_values=np.nan,strategy='mean')), ('minmax_scaler',MinMaxScaler())]) 
            clf.fit(data)
            joblib.dump(clf,i+'.dataProcessing')
    else:
        clf=joblib.load(i+'.dataProcessing')
    return clf.transform(data)

def cal_var(features,i,istrain=0,var=0):
    if istrain==1:
        if os.path.exists(i+'.delvar0'):
            sel=joblib.load(i+'.delvar0')
            sel.transform(features)
        else:
            sel = VarianceThreshold(threshold=(var))
            sel.fit(features)
            joblib.dump(sel,i+'.delvar0')    
    else:
        sel=joblib.load(i+'.delvar0')
    features=pd.DataFrame(sel.transform(features), columns=list(sel.get_feature_names_out()))
    return features
