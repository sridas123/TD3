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
DDPG_pretrain=[]
DDPG_bc=[]
DDPG_bc_pretrain=[]
DDPG_bc_l2=[]
step=5000
for i in range(0,no_of_runs):
    DDPG.append(np.load(result_dir+filename+"_"+str(i)+".npy"))
    DDPG[i]=DDPG[i][0:401]
    
for i in range(0,no_of_runs):
    DDPG_pretrain.append(np.load(result_dir+filename+"_"+str(i)+"_demo.npy"))

for i in range(0,no_of_runs):
    DDPG_bc.append(np.load(result_dir+filename+"_"+str(i)+"_demo_bc.npy"))
    
for i in range(0,no_of_runs):
    DDPG_bc_l2.append(np.load(result_dir+filename+"_"+str(i)+"_demo_bc_l2.npy"))
    
for i in range(0,no_of_runs):
    DDPG_bc_pretrain.append(np.load(result_dir+filename+"_"+str(i)+"_demo_bc_pretrain.npy"))
    
#print (DDPG[0].shape,DDPG[1].shape)
#print (DDPG_demo[0].shape,DDPG_demo[1].shape)
DDPG_mean=np.mean(DDPG,axis=0)
#print (DDPG_mean.shape)
DDPG_std=np.std(DDPG,axis=0)
#print (DDPG_std)
DDPG_pretrain_mean=np.mean(DDPG_pretrain,axis=0)
DDPG_pretrain_std=np.std(DDPG_pretrain,axis=0)

DDPG_bc_mean=np.mean(DDPG_bc,axis=0)
DDPG_bc_std=np.std(DDPG_bc,axis=0)

DDPG_bc_l2_mean=np.mean(DDPG_bc_l2,axis=0)
DDPG_bc_l2_std=np.std(DDPG_bc_l2,axis=0)

DDPG_bc_pretrain_mean=np.mean(DDPG_bc_pretrain,axis=0)
DDPG_bc_pretrain_std=np.std(DDPG_bc_pretrain,axis=0)

#Plot the evaluation reward plot
itertn=list(range(0,DDPG_mean.shape[0]*5000,step))
print (len(itertn))
"""Blue color is for suggested algorithm accuracy"""
plt.plot(itertn,DDPG_mean,color='b',label='DDPG')
#plt.plot(itertn,DDPG_pretrain_mean,color='r',label='DDPG_D_pretrain')
plt.plot(itertn,DDPG_bc_mean,color='g',label='DDPG_D_BC')
plt.plot(itertn,DDPG_bc_l2_mean,color='m',label='DDPG_D_BC+L2')
plt.plot(itertn,DDPG_bc_pretrain_mean,color='c',label='DDPG_D_BC+pretrain')
plt.ylim(-800,300)
#plt.errorbar(itertn,DDPG_mean,yerr=DDPG_std,color='b',label='DDPG')
#plt.errorbar(itertn,DDPG_demo_mean,yerr=DDPG_demo_std,color='r',label='DDPG_FD_pretrain')
#plt.xlim(0,itertn)
#plt.ylim(0.5,1)
plt.xlabel('No. of time-steps')
plt.ylabel(' Average evaluation reward')
plt.legend()
plt.savefig(plot_dir+'evaluation_reward_llandar_minus_pretrain.png', bbox_inches='tight')
plt.clf()