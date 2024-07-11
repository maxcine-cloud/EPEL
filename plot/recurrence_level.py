# encoding=utf-8
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

color=['#0000FF','#0000FF'] 
label = ['PRE','SEN','SPE','F1-score','ACC','AUC','AUPR']
font_size=25
plt.figure(figsize = (18,4))
    

font5 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}    

def recurrence(data, dataPath):   
    x = ['2','3','4','5','6','7']
    plt.subplot(1,2,1)
    for i in range(0,1): 
        tem=data.iloc[:,i+1].str.rsplit('±',expand=True)
        tem=tem.astype('float')
        y = tem[0].tolist()
        
        upper_limit = tem[1].tolist()
        plt.errorbar(x, y, yerr=[upper_limit, upper_limit], c=color[i], fmt='--',capsize=5,ecolor=color[i],label=label[i]) #fmt='.',,mec=color[i]mfc=color[i],
        plt.xlabel('Recurrence Level',font5)
        plt.ylabel('AUC',font5)

        plt.xticks(size = font_size,family= 'Times New Roman')
        plt.yticks(np.arange(0.60,1.00,0.08),family= 'Times New Roman',size = font_size)

        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        
    plt.subplot(1,2,2)
    for i in range(1,2): 
        tem=data.iloc[:,i+1].str.rsplit('±',expand=True)
        tem=tem.astype('float')
        y = tem[0].tolist()
        
        upper_limit = tem[1].tolist()
        plt.errorbar(x, y, yerr=[upper_limit, upper_limit], c=color[i], fmt='--',capsize=5,ecolor=color[i],label=label[i]) #fmt='.',,mec=color[i]mfc=color[i],
        plt.xlabel('Recurrence Level',font5)
        plt.ylabel('AUPR',font5)
        
        plt.xticks(size = font_size,family= 'Times New Roman')
        plt.yticks(np.arange(0.65,1.00,0.08),family= 'Times New Roman',size = font_size)

        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    plt.gcf().patch.set_alpha(0)
    plt.savefig(dataPath+"/figures/epSMS_r"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2,transparent=True)

data_path=os.path.split(os.path.realpath(__file__))[0]
data=pd.read_csv(data_path+'/recurrence_level.txt',sep='\t')
recurrence(data, data_path)