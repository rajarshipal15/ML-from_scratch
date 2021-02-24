#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:13:08 2021

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



file='/media/rajarshi/My Passport/Home/codes/Ng-ml/Andrew-Ng-ML-Course-Assignments-master/machine-learning-ex8/ex8/ex8data2.mat'
tmp1=scipy.io.loadmat(file)
X,Xval,yval=tmp1['X'],tmp1['Xval'],tmp1['yval']
m,n=len(X[:,0]), len(X[0,:])


def Gaussian(X,Mu,Sigma):
    m,n=len(X[:,0]), len(X[0,:])
    assert len(Mu)==n and len(Sigma)==n, "Shape error for means and sds"
    tmp1=1.0
    for i in range(n): 
        tmp=(X[:,i]-Mu[i])/Sigma[i]
        tmp1=tmp1*np.exp(-tmp*tmp/2.0)/(np.sqrt(2.0*np.pi)*Sigma[i])
    return tmp1

Mu, Sigma=np.zeros(n),np.zeros(n)
for i in range(n):
    Mu[i],Sigma[i]=np.mean(X[:,i]),np.sqrt(np.var(X[:,i],ddof=1))

pval=Gaussian(Xval,Mu,Sigma)
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
pval1=Gaussian(X,Mu,Sigma)
pos1=np.where(pval1 < bestEpsilon)[0]
print("The threshold obtained for the high dimesional dataset is",bestEpsilon)
print("Number of anomalies detected equal to",len(pos1))
    