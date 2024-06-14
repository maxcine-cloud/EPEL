# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def fold2(moduleDT,X,y,CV=10):

    skflods = StratifiedKFold(n_splits=CV,shuffle=False)
    a=[]
    y_pred_prob_all=np.zeros((X.shape[0],2))
    for train_index,test_index in skflods.split(X,y):
        clone_clf =clone(moduleDT)
        X_sub=np.array(X)
        y_sub=np.array(y)
        X_train_folds = X_sub[train_index]
        y_train_folds = y_sub[train_index]
        X_test_folds = X_sub[test_index]
        y_test_folds = y_sub[test_index]
        clone_clf.fit(X_train_folds,y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        y_pred_prob = clone_clf.predict_proba(X_test_folds)
        y_pred_prob_all[test_index]=y_pred_prob
        [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]=metr1_2(y_test_folds,y_pred,y_pred_prob[:,1])
        a.append([np.round(Pre,3),np.round(recall,3),np.round(specificity,3),np.round(BACC,3),np.round(F1_Score,3),np.round(MCC,3),np.round(ACC,3),np.round(roc_auc,3),np.round(PRC,3)])
        # a.append([Pre,recall,specificity,BACC,F1_Score,MCC,ACC,roc_auc,PRC])
    return a,y_pred_prob_all

def fold_only(y,y_pred,y_prob,CV=10):
    skflods = StratifiedKFold(n_splits=CV,shuffle=False)
    a=[]
    for train_index,test_index in skflods.split(y_prob,y):
        y_prob_sub=np.array(y_prob)
        y_pred_sub=np.array(y_pred)
        y_sub=np.array(y)
        prob = y_prob_sub[test_index]
        label = y_sub[test_index]
        pred = y_pred_sub[test_index]
        [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]=metr1_2(label,pred,prob)
        a.append([np.round(Pre,3),np.round(recall,3),np.round(specificity,3),np.round(BACC,3),np.round(F1_Score,3),np.round(MCC,3),np.round(ACC,3),np.round(roc_auc,3),np.round(PRC,3)])
    return a


def metr1_2(y_test,pred,pred_prob):
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
    ACC=metrics.accuracy_score(y_test,pred)
    MCC=metrics.matthews_corrcoef(y_test,pred)
    recall=metrics.recall_score(y_test, pred)
    Pre = metrics.precision_score(y_test, pred)
    specificity=tn/(tn+fp)
    BACC=(recall+specificity)/2.0
    F1_Score=metrics.f1_score(y_test, pred)
    fpr,tpr,threshold=metrics.roc_curve(y_test,pred_prob)
    roc_auc=metrics.auc(fpr,tpr)
    precision_prc, recall_prc, _ = metrics.precision_recall_curve(y_test, pred_prob)
    PRC = metrics.auc(recall_prc, precision_prc)
    return  [tn, fp, fn, tp,ACC,Pre,recall,MCC,specificity,BACC,F1_Score,roc_auc,PRC]

def cal_mean_var(kfold_metrics,names):
    """
    kfold_metrics:[5*n,10,9]
    names:[,2]
    """
    mean=pd.DataFrame(np.round(np.mean(np.array(kfold_metrics).reshape(-1,10,9),axis=1),3),columns=['PRE','SEN','SPE','BACC','F1-score','MCC','ACC','AUC','AUPR'])
    std=pd.DataFrame(np.round(np.std(np.array(kfold_metrics).reshape(-1,10,9),axis=1),3),columns=['PRE','SEN','SPE','BACC','F1-score','MCC','ACC','AUC','AUPR'])
    mean=mean.astype(str)
    std=std.astype(str)
    all_metric=pd.DataFrame(np.array(names).reshape(-1,2))
    for me_na,clona in zip(['PRE','SEN','SPE','BACC','F1-score','MCC','ACC','AUC','AUPR'],list(mean.columns)):
        all_metric[me_na]=mean[clona]+'±'+std[clona]
    return all_metric
