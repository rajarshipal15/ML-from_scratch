#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:41:22 2021

@author: rajarshi
"""


import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as opt
from numpy import meshgrid
import scipy.io

file='/media/rajarshi/My Passport/Home/codes/Ng-ml/Ex4/ex4data1.mat'
file1='/media/rajarshi/My Passport/Home/codes/Ng-ml/Ex4/ex4weights.mat'
tmp=scipy.io.loadmat(file1)

Theta1, Theta2=tmp['Theta1'],tmp['Theta2']

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


tmp1=scipy.io.loadmat(file)
X,y=tmp1['X'],tmp1['y'] 
#y[y > 9.0]=0.0
m=len(X[:,0])
y=y.reshape(m,)
X=np.concatenate((np.reshape(np.ones(5000),(5000,1)),X),axis=1)

#Data manipulation to take care of matlab indexing
y=[i-1 for i in y]
ytmp,Xtmp=y[0:500],X[0:500,:]
ytmp1,Xtmp1=np.delete(y,range(500)),np.delete(X,range(500),axis=0)
y,X=np.append(ytmp1,ytmp),np.vstack((Xtmp1,Xtmp))


tmp3=0

for j in range(m):
#    tmpa=X[j,:]
#    tmpb= np.append(1,sigmoid(Theta1 @ X[j,:]))
    tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[j,:])))
    if (tmpc.argmax()  == int(y[j])):
        tmp3 +=1
ACTRS=(tmp3/m)*100 

K=10
yp,xp=np.zeros((m,K)),np.zeros((m,K))
for i in range(m):
    temp=np.zeros(K)
    temp[y[i]]=1.0
    yp[i,:]=temp
    tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[i,:])))
    xp[i,:]=tmpc

#Cost without regularisation 
lamb=1.0
Cost=(1.0/m)*np.sum(-yp*np.log(xp)-(1.0-yp)*np.log(1.0-xp))
print("The cost obtained without regularisation for the pre-trained network is",Cost)
#Cost obtained from .287629

Costr= Cost + (lamb/(2.0 *m))*(np.sum(Theta1[:,1:]*Theta1[:,1:])+ np.sum(Theta2[:,1:]*Theta2[:,1:]))
#Cost 0.383769
print("The cost obtained with regularisation with the pre-trained network is",Costr)

