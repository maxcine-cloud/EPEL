# encoding=utf-8
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

font5 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

def distrbu(scores):
    plt.figure(figsize = (9,4))
    font_size=20
    hist, bin_edges = np.histogram(scores, bins=50, density=False)
    percentages = hist / len(scores) 
    plt.bar(bin_edges[:-1], percentages, width=0.5*(bin_edges[1]-bin_edges[0]))
    plt.axhline(y=0.02, color='gray', linestyle='--')

    plt.xlabel('sSNVs Effect Scores for EPEL',font5)
    plt.ylabel('Proportion',font5)
    plt.xticks(np.arange(0,1.1,0.1),size = font_size,family= 'Times New Roman')
    plt.yticks(np.arange(0,0.2,0.02),family= 'Times New Roman',size = font_size)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlim([0,1])
    plt.ylim([0,0.1])
    plt.yticks(size = 15,family= 'Times New Roman')
    plt.gcf().patch.set_alpha(0)
    
    plt.savefig(data_path+"/epSMS_whole_effectScore"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2,transparent=True)

data_path=os.path.split(os.path.realpath(__file__))[0]
scores=pd.read_csv(data_path+'/GRCh37_whole_genome_sSNVs_EPEL.vcf').iloc[:,4]
distrbu(scores)