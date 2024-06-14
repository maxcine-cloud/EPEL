# -*- coding: utf-8 -*-
import pandas as pd

def imfile(path1,hashead=None):
    if hashead:
        left=pd.read_csv(path1,sep='\t')
    else:               
        left=pd.read_csv(path1,sep='\t',header=None)
    return left

def tofile(data,path1,tohead=None):
    if tohead:
        data.to_csv(path1,sep='\t',index=False)
    else:               
        data.to_csv(path1,sep='\t',index=False,header=False)
    return data
