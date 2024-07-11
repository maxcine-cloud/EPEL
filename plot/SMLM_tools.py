# encoding=utf-8
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


font_size=25

def SMLM_tool(data, data_path):

 
    fig, ax = plt.subplots(figsize=(13,6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    width = 0.2
    index = np.arange(len(categories))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 25
    
    categories = ['SPE', 'PRE', 'SEN', 'F1-score', 'MCC','ACC', 'AUC', 'AUPR'] 
    tool_names=['DNABERT','SMLM1','SMLM2','EPEL']
    color=['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2']

    for i in range(0,4):
        tem=data.iloc[:,i+1].str.rsplit('±',expand=True)
        tem=tem.astype('float')
        tool_name = tem[0].tolist()
        tool_bar = tem[1].tolist()
        ax.bar(index - 1.5*width,tool_name , width, label=tool_names[i], color=color[i], yerr=tool_bar)

    ax.legend(prop={'family': 'Times New Roman', 'size': 20}, ncol=2,loc='upper right',labelspacing=0.1,columnspacing=0.4,frameon=False,bbox_to_anchor=(1, 1.10))
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    plt.ylim([0, 1])

    plt.xticks()

    plt.gcf().patch.set_alpha(0)
    plt.savefig(data_path+"/SMLM_4tools"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2,transparent=True)
    plt.close(0)

data_path=os.path.split(os.path.realpath(__file__))[0]
data=pd.read_csv(data_path+'/SMLM_tools.txt',sep='\t')
SMLM_tool(data, data_path)