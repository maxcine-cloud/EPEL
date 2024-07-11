# encoding=utf-8
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from itertools import cycle
from sklearn.metrics import roc_curve, auc, precision_recall_curve

color=['#0000FF','#0000FF'] 
label = ['PRE','SEN','SPE','F1-score','ACC','AUC','AUPR']
font_size=25
plt.figure(figsize = (18,4))


font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}

font6 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 37,
}

def draw_roc(data, name,outpathway):
    data, nameList = data[:,:3],data[:,3]
    plt.subplot(1,2,1)
    colors = cycle(['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']) 
    
    for i,(mol,color) in enumerate(zip(data,colors)):
        plt.plot(mol[0], mol[1], color=color,
                      lw=3 if i!=0 else 3, label=nameList[i]+' (AUC = %0.3f)' % mol[2], linestyle='dashdot' if i%2!=0 else '-')
    
    plt.xticks(size = 40,family= 'Times New Roman')
    plt.yticks(size = 40,family= 'Times New Roman')

    plt.xlim([-0.025, 1.075])
    plt.ylim([-0.025, 1.025])

    plt.xlabel('False Positive Rate',font3)
    plt.ylabel('True Positive Rate',font3)
    
    legend = plt.legend(loc="lower right",ncol= 1,prop=font6,columnspacing=0.2,labelspacing=0.2,frameon=False)  #lower right font4
    legend.get_title().set_fontsize(fontsize = 'small')


def draw_pr(data, name,outpathway):
    data, nameList = data[:,:3],data[:,3]
    plt.subplot(1,2,2)
    
    colors = cycle(['#050f2c', 'goldenrod', '#00aeff', '#8e43e7', 'steelblue', '#ff6c5f', '#ffc168', '#2dde98','y','lightsteelblue','m','green','tan','blue','yellow']) 
    
    for i,(mol,color) in enumerate(zip(data,colors)):
        plt.plot(mol[1], mol[0], color=color,
                      lw=3 if i!=0 else 3, label=nameList[i]+' (AUPR = %0.3f)' % mol[2], linestyle='dashdot' if i%2!=0 else '-')
   
    plt.xticks(size = 40,family= 'Times New Roman')
    plt.yticks(size = 40,family= 'Times New Roman')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xlabel('Recall',font3) 
   
    plt.ylabel('Precision',font3)

    
    legend = plt.legend(loc="lower right",ncol= 1,prop=font6,columnspacing=0.2,labelspacing=0.2,frameon=False) #font4
    legend.get_title().set_fontsize('large')   
    
    plt.savefig(outpathway+"/au_cosmicTest2"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)
    plt.close(0)

data_path=os.path.split(os.path.realpath(__file__))[0]

toolsnameList = ['CS','CSS','epSMic','EPEL']


new_data1=pd.read_csv(data_path+'/cosmicTest_4_tools_dulchongfu.txt',hashead=1)
new_data1=new_data1[['chr','pos','ref','alt','CScape','CScape-somatic','epSMic','epBERT','label_x']]
new_data=[]

for i in range(4,9):
    new_data.append(new_data1.iloc[:,i,].tolist())

new_data=np.array(new_data)
index=[0,1,2,3]
aucinputdata = []
praucinputdata = []

for i, other in enumerate(new_data[index]):

    y=new_data[4].reshape(-1,1)
    x=other.reshape(-1,1)
    sub=np.column_stack((y, x))
    sub[sub==''] = np.nan
    sub=np.array(sub,dtype='float32')
    print(sub.shape)
    sub=sub[~np.isnan(sub).any(axis=1), :]
    print(sub.shape)

    fpr, tpr, thresholds = roc_curve(sub[:,0],sub[:,1])
    _auc = auc(fpr, tpr)
    aucinputdata.append([fpr, tpr,_auc]) 
    precision, recall, _thresholds = precision_recall_curve(sub[:,0],sub[:,1])
    _prauc = auc(recall, precision)
    praucinputdata.append([precision, recall,_prauc])

newdata = np.column_stack([aucinputdata, toolsnameList])
plt.figure(figsize = (30,15))
draw_roc(newdata, type,data_path)
prnewdata = np.column_stack([praucinputdata, toolsnameList])
draw_pr(prnewdata, type,data_path) 

