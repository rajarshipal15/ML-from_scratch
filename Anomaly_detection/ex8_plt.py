#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:49:20 2021

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

X=tmp1['X']
m,n=len(X[:,0]), len(X[0,:])
plt.plot(X[:,0],X[:,1],'.')
plt.xlabel("Latency(ms)")
plt.ylabel("Throughput(mb/sec)")
Mu, Sigma=np.zeros(n),np.zeros(n)

for i in range(n):
    Mu[i],Sigma[i]=np.mean(X[:,i]),np.std(X[:,i])

delta=0.025
xrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), delta)
yrange = np.arange(np.min(X[:,0]), np.max(X[:,0]), delta)
X1, Y1 = meshgrid(xrange,yrange)    
Xp,Yp=(X1-Mu[0])/(Sigma[0]), (Y1-Mu[1])/(Sigma[1])
Z=np.exp(-Xp*Xp/2.0 - Yp*Yp/2.0)
plt.contour(X1, Y1,Z, np.linspace(.05,1.0,10))    
