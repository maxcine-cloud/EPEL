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
    

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24}


data_path=os.path.split(os.path.realpath(__file__))[0]

fs_index=pd.read_csv(data_path+'/fs_index.txt')
fs_name=pd.read_csv(data_path+'/fs_name.txt',header=None)

fs_name=fs_name.astype(str)
fs_name['union']=fs_name[1]+' ('+fs_name[0]+')'
Catboost_GBDT=fs_index['gbdt'][:12]
RF_XGB=fs_index['xgb'][:47]
GBDT_GBDT=fs_index['gbdt'][:12]
AB_XGB=fs_index['xgb'][:4]
XGB_XGB=fs_index['xgb'][:47]

res=[]


for name in [Catboost_GBDT,RF_XGB,GBDT_GBDT,AB_XGB,XGB_XGB]:
    fea={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0,'19':0,
        '20':0,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0,'27':0,'28':0,'29':0,'30':0,'31':0,'37':0,'43':0,'49':0,'54':0,'59':0,'64':0,'69':0,'74':0,'77':0,'81':0,
        '85':0,'90':0,'106':0,'122':0,'138':0}
    for i in name:

        try:
            fea[str(i)]+=1
        except:
            if i in list(range(32,37)):
                fea['37']+=1
            elif i in list(range(38,43)):
                fea['43']+=1
            elif i in list(range(44,49)):
                fea['49']+=1
            elif i in list(range(50,54)):
                fea['54']+=1
            elif i in list(range(55,59)):
                fea['59']+=1
            elif i in list(range(60,64)):
                fea['64']+=1                       
            elif i in list(range(65,69)):
                fea['69']+=1
            elif i in list(range(70,74)):
                fea['74']+=1
            elif i in list(range(75,77)):
                fea['77']+=1   
            elif i in list(range(78,81)):
                fea['81']+=1  
            elif i in list(range(82,85)):
                fea['85']+=1  
            elif i in list(range(86,90)):
                fea['90']+=1  
            elif i in list(range(91,106)):
                fea['106']+=1  
            elif i in list(range(107,122)):
                fea['122']+=1  
            elif i in list(range(123,138)):
                fea['138']+=1      
    for key,num in zip([37,43,49,54,59,64,69,74,77,81,85,90,106,122,138],[6,6,6,5,5,5,5,5,3,4,4,5,16,16,16]):
        fea[str(key)]=fea[str(key)]/num            
    res.append(list(fea.values()))            

res2=np.array(res)                                   
                      
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

plt.rc('font', **font)
plt.figure(figsize=(23, 2))
plt.imshow(res2, interpolation='none', cmap='Blues')
plt.xticks(range(len(fea.keys())), fs_name['union'].tolist(),rotation=90,size = 20,family= 'Times New Roman')
plt.yticks(range(5), ['CatBoost-GIS','RF-XIS','GBDT-GIS','AdaBoost-XIS','XGB-XIS'],size = 20,family= 'Times New Roman')

cax = plt.gca().inset_axes([-0.12, -0.6, 0.1, 0.05])
cbar = plt.colorbar(cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)

plt.savefig(data_path+"/fs_map_cls5"+".png",dpi=600,bbox_inches="tight", pad_inches=0.2)


