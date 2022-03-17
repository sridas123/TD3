#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:46:04 2022

@author: srijita
"""
import numpy as np
import matplotlib.pyplot as plt

result_dir="../../results/"
plot_dir="../../plots/"
filename="OurTD3_LunarLanderContinuous-v2"
no_of_runs=3
DDPG=[]
DDPG_demo=[]
step=5000
for i in range(0,no_of_runs):
    DDPG.append(np.load(result_dir+filename+"_"+str(i)+".npy"))
    DDPG[i]=DDPG[i][0:401]
    
for i in range(0,no_of_runs):
    DDPG_demo.append(np.load(result_dir+filename+"_"+str(i)+"_demo.npy"))
    
#print (DDPG[0].shape,DDPG[1].shape)
#print (DDPG_demo[0].shape,DDPG_demo[1].shape)
DDPG_mean=np.mean(DDPG,axis=0)
#print (DDPG_mean.shape)
DDPG_std=np.std(DDPG,axis=0)
#print (DDPG_std)
DDPG_demo_mean=np.mean(DDPG_demo,axis=0)
DDPG_demo_std=np.std(DDPG_demo,axis=0)

#Plot the evaluation reward plot
itertn=list(range(0,DDPG_mean.shape[0]*5000,step))
print (len(itertn))
"""Blue color is for suggested algorithm accuracy"""
plt.plot(itertn,DDPG_mean,color='b',label='DDPG')
plt.plot(itertn,DDPG_demo_mean,color='r',label='DDPG_FD_pretrain')
#plt.errorbar(itertn,DDPG_mean,yerr=DDPG_std,color='b',label='DDPG')
#plt.errorbar(itertn,DDPG_demo_mean,yerr=DDPG_demo_std,color='r',label='DDPG_FD_pretrain')
#plt.xlim(0,itertn)
#plt.ylim(0.5,1)
plt.xlabel('No. of time-steps')
plt.ylabel(' Average evaluation reward')
plt.legend()
plt.savefig(plot_dir+'evaluation_reward_llandar.png', bbox_inches='tight')
plt.clf()