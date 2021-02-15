#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:17:32 2021

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

#Each movie is like a training example and you are learning theta for each user by a linear reg.
# Y, R are n_mx n_u matrices
file='ex8_movies.mat'
file1='ex8_movieParams.mat'
tmp1=scipy.io.loadmat(file)
Y,R = tmp1['Y'],tmp1['R']
n_m,n_u=len(Y[:,0]),len(Y[0,:])
tmp2=scipy.io.loadmat(file1)
X,Theta,num_users,num_movies,num_features=tmp2['X'],tmp2['Theta'],tmp2['num_users'],tmp2['num_movies'],tmp2['num_features']
num_users = 4; num_movies = 5; num_features = 3
X=X[:num_movies,:num_features]
Theta=Theta[:num_users,:num_features]
Y,R=Y[:num_movies,:num_users], R[:num_movies,:num_users]
Predicted_ratings= (X @ Theta.T)

tmp= R*(Y- Predicted_ratings)

Cost=0.5*np.sum(tmp*tmp)

f= lambda a,b : np.concatenate((a.flatten(),b.flatten()))
n=num_movies*num_features + num_users*num_features
def Costfunc(var):
    n=num_movies*num_features + num_users*num_features
    X=var[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
    Theta=var[num_movies*num_features:n ].reshape((num_users,num_features))
    Predicted_ratings= (X @ Theta.T)
    tmp= R*(Y- Predicted_ratings) 
    return  0.5*np.sum(tmp*tmp)

def Grad(var):
    n=num_movies*num_features + num_users*num_features
    X=var[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
    Theta=var[num_movies*num_features:n ].reshape((num_users,num_features))
    #Vectorized  code for gradient wrt X
    Gradx,Gradtheta=np.zeros_like(X), np.zeros_like(Theta)   
    for i in range(num_movies):
        idx=np.where(R[i,:]==1)[0]
        tmp=(X @ Theta.T)-Y
        Gradx[i,:]=tmp[i,idx] @ Theta[idx,:]
    #Vectorized code for gradient wrt theta
    for j in range(num_users):
        idx=np.where(R[:,j]==1)[0] 
        tmp=((X @ Theta.T)-Y).T
        Gradtheta[j,:]=tmp[j,idx] @ X[idx,:]
    return f(Gradx,Gradtheta)

eps=0.01
gradtheta=np.zeros(n)
thetaini=np.random.uniform(-3,3,n)
#thetaini=np.zeros(n)
for i in range(n):
#    thetap,thetam=np.zeros(n),np.zeros(n)
    thetap=np.asarray([k for k in thetaini])
    thetam=np.asarray([k for k in thetaini])
    thetap[i],thetam[i]=thetap[i]+ eps, thetam[i]-eps     
    gradtheta[i]=(Costfunc(thetap)-Costfunc(thetam))/(2.0*eps)

temp=np.max(abs(Grad(thetaini)-gradtheta))





