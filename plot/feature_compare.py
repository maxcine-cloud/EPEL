# encoding=utf-8
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

color=['#FF4B5C','#86C4BA'] 
label = ['AUC','AUPR']
font_size=23
def featureGroups(data,dataPath):
    
    plt.figure(figsize = (9,4))
    x = ['w/o DNAS', 'w/o PCS', 'w/o CBERT', 'w/o (DNAS\n+PCS\n+CBERT)', 'EPEL'] 

    for i in range(0,2): 
        tem=data.iloc[:,i+1]
        tem=tem.astype('float')
        y = tem
        
        plt.plot(x, y,label=label[i],marker='o')
        plt.axhline(y=0.871, color='gray', linestyle='--')
        plt.axhline(y=0.894, color='gray', linestyle='--')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),frameon=False,fontsize=25,columnspacing=0.4,labelspacing=0.4,ncol=2)#,prop=font5
        
        plt.xticks(size = font_size,family= 'Times New Roman',rotation=45)
        plt.yticks(np.arange(0.86,0.90,0.01),family= 'Times New Roman',size = font_size)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    plt.gcf().patch.set_alpha(0)
    plt.savefig(dataPath+"/epSMS_featureGroups"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2,transparent=True)

data_path=os.path.split(os.path.realpath(__file__))[0]
data=pd.read_csv(data_path+'/feature_groups.txt',sep='\t')
featureGroups(data, data_path)