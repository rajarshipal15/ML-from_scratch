#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:18:53 2021

@author: rajarshi
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy import stats
import scipy.optimize as opt
from numpy import meshgrid
import scipy.io
from matplotlib import image
import sys

file='/media/rajarshi/My Passport/Home/codes/Ng-ml/Andrew-Ng-ML-Course-Assignments-master/machine-learning-ex8/ex8/ex8data1.mat'
tmp1=scipy.io.loadmat(file)
X,Xval,yval=tmp1['X'],tmp1['Xval'],tmp1['yval']
m,n=len(X[:,0]), len(X[0,:])

Mu, Sigma=np.zeros(n),np.zeros(n)

# for i in range(n):
#     Mu[i],Sigma[i]=np.mean(X[:,i]),np.std(X[:,i])
for i in range(n):
    Mu[i],Sigma[i]=np.mean(X[:,i]),np.sqrt(np.var(X[:,i],ddof=1))

#Plot**********************************************
plt.plot(X[:,0],X[:,1],'.')
plt.xlabel("Latency(ms)")
plt.ylabel("Throughput(mb/sec)")
delta=0.025
xrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), delta)
yrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), delta)
X1, Y1 = meshgrid(xrange,yrange)    
Xp,Yp=(X1-Mu[0])/(Sigma[0]), (Y1-Mu[1])/(Sigma[1])
Z=np.exp(-Xp*Xp/2.0 - Yp*Yp/2.0)/(2.0*np.pi*Sigma[0]*Sigma[1])
plt.contour(X1, Y1,Z, np.linspace(.001,.1,10))    
#Plot********************************************

#Normalisation Errors........*************


Xp,Yp=(Xval[:,0]-Mu[0])/(Sigma[0]), (Xval[:,1]-Mu[1])/(Sigma[1])

pval=np.exp(-Xp*Xp/2.0 -Yp*Yp/2.0 )/(2.0*np.pi*Sigma[0]*Sigma[1])
bestF1,epsilon=0.0,0.0
for epsilon in  np.linspace(np.min(pval),np.max(pval),1000)[1:]:
    pos=np.where(pval < epsilon)[0]
    neg=np.where(pval >= epsilon)[0]
    tp=len(np.where(yval[pos,0] ==1)[0])
    fp=len(np.where(yval[pos,0] ==0)[0])
    fn=len(np.where(yval[neg,0] ==1)[0])
    prec=tp/(tp+fp)
    rec=tp/(tp+fn)
    F1=(2*prec*rec)/(prec+rec)
    if (F1 > bestF1):
        bestF1=F1
        bestEpsilon=epsilon

#bestF1
#Out[54]: 0.8750000000000001  
#bestEpsilon
#Out[53]: 0.00018419324561013233

Xp1,Yp1=(X[:,0]-Mu[0])/(Sigma[0]), (X[:,1]-Mu[1])/(Sigma[1])
pval1=np.exp(-Xp1*Xp1/2.0 -Yp1*Yp1/2.0 )/(2.0*np.pi*Sigma[0]*Sigma[1])
pos1=np.where(pval1 < bestEpsilon)[0]
plt.plot(X[:,0][pos1],X[:,1][pos1],'x',label='Anomalies')
plt.legend(loc=0)

print("The best threshold obtained is ", bestEpsilon)



