#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:22:42 2021

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
from PIL import Image
import scipy.io

file='ex7faces.mat'

#
tmp=scipy.io.loadmat(file)
X=tmp['X']
m,n=len(X[:,0]), len(X[0,:])
# fig, axs = plt.subplots(10, 10)
# for i in range(10):
#     for j in range(10):
#         tmp2=(X[10*i +j ,:].reshape(32,32)).T
#         axs[i,j].imshow(tmp2, cmap='gray_r') 
#         axs[i,j].axis('off') 

def scal(x):
    return (x-np.mean(x)*np.ones(len(x)))/np.std(x)    

for j in range(n):  
    X[:,j]=scal(X[:,j])
    
Sigma=(X.T @ X)/m

U=scipy.linalg.svd(Sigma)[0]  
K=100
U_r=U[:,:K]

# fig, axs = plt.subplots(6, 6)
# for i in range(6):
#     for j in range(6):
#         tmp2=(U_r.T[6*i +j ,:].reshape(32,32)).T
#         axs[i,j].imshow(tmp2, cmap='gray_r') 
#         axs[i,j].axis('off') 

Z=np.zeros((m,K))
for j in range(m):
    Z[j,:]=U_r.T @ X[j,:]

X_ap=np.zeros_like(X)    
for j in range(m):
    X_ap[j,:]=U_r @ Z[j,:]    
 
fig1, axs1 = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):
        tmp2=(X_ap[10*i +j ,:].reshape(32,32)).T
        axs1[i,j].imshow(tmp2, cmap='gray_r') 
        axs1[i,j].axis('off') 
    