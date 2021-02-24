#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 18:57:43 2021

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

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigGrad(z):
    return sigmoid(z)*(1.0-sigmoid(z))
tmp1=scipy.io.loadmat(file)
X,y=tmp1['X'],tmp1['y'] 
#y[y > 9.0]=0.0
m=len(X[:,0])
y=y.reshape(m,)
X=np.concatenate((np.reshape(np.ones(5000),(5000,1)),X),axis=1)

f= lambda a,b : np.concatenate((a.flatten(),b.flatten()))
#Data manipulation to take care of matlab indexing
y=[i-1 for i in y]
ytmp,Xtmp=y[0:500],X[0:500,:]
ytmp1,Xtmp1=np.delete(y,range(500)),np.delete(X,range(500),axis=0)
y,X=np.append(ytmp1,ytmp),np.vstack((Xtmp1,Xtmp))
###################################################

Theta1,Theta2=np.load('Theta1f-400it.npy'), np.load('Theta2f-400it.npy')


tmp3=0

for j in range(m):
#    tmpa=X[j,:]
#    tmpb= np.append(1,sigmoid(Theta1 @ X[j,:]))
    tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[j,:])))
    if (tmpc.argmax()  == int(y[j])):
        tmp3 +=1
ACTRS=(tmp3/m)*100 

print("The accuracy obtained for identification is",ACTRS)
#ACTRS=99.4 for 400 it


