#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:19:21 2021

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
#X,Theta,num_users,num_movies,num_features=tmp2['X'],tmp2['Theta'],tmp2['num_users'],tmp2['num_movies'],tmp2['num_features']

num_users = 200; num_movies = 500; num_features = 10

# X=X[:num_movies,:num_features]
# Theta=Theta[:num_users,:num_features]
Y,R=Y[:num_movies,:num_users], R[:num_movies,:num_users]
# Predicted_ratings= (X @ Theta.T)

# tmp= R*(Y- Predicted_ratings)

# Cost=0.5*np.sum(tmp*tmp)

f= lambda a,b : np.concatenate((a.flatten(),b.flatten()))
n=num_movies*num_features + num_users*num_features
# Regularisation parameter
lamb=10.0
#Initiliasation of X and Theta
varini=np.random.normal(loc=0.0,scale=1.0,size=n)
X_ini=varini[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
Theta_ini=varini[num_movies*num_features:n ].reshape((num_users,num_features))

Mu=np.zeros(len(Y[:,0]))
for j in range(len(Y[:,0])):
    idx=np.where(R[j,:]==1)[0]
    Mu[j]=np.mean(Y[j,idx])


def Normalize_ratings(Y,Mu):
    Yp=np.zeros((num_movies,num_users))
    for i in range(len(Y[:,0])):
        idx=np.where(R[i,:]==1)[0]
        Yp[i,idx]=Y[i,idx]-Mu[i] 
    return Yp

Y1=Normalize_ratings(Y,Mu)



def Costfunc(var):
    n=num_movies*num_features + num_users*num_features
    X=var[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
    Theta=var[num_movies*num_features:n ].reshape((num_users,num_features))
    Predicted_ratings= (X @ Theta.T)
    tmp= R*(Y1- Predicted_ratings) 
    return  0.5*np.sum(tmp*tmp) + 0.5*lamb*(np.sum(X*X) + np.sum(Theta*Theta))

def Grad(var):
    n=num_movies*num_features + num_users*num_features
    X=var[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
    Theta=var[num_movies*num_features:n ].reshape((num_users,num_features))
    #Vectorized  code for gradient wrt X
    Gradx,Gradtheta=np.zeros_like(X), np.zeros_like(Theta)   
    for i in range(num_movies):
        idx=np.where(R[i,:]==1)[0]
        tmp=(X @ Theta.T)-Y1
        Gradx[i,:]=tmp[i,idx] @ Theta[idx,:] + lamb*X[i,:]
    #Vectorized code for gradient wrt theta
    for j in range(num_users):
        idx=np.where(R[:,j]==1)[0] 
        tmp=((X @ Theta.T)-Y1).T
        Gradtheta[j,:]=tmp[j,idx] @ X[idx,:] + lamb*Theta[j,:]
    return f(Gradx,Gradtheta)




res = opt.minimize(Costfunc, varini, method='BFGS', jac=Grad,options={'disp': True,'maxiter': 100})
X=res.x[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
Theta=res.x[num_movies*num_features:n ].reshape((num_users,num_features))

Pr_rat= (X @ Theta.T)
Ynew=np.zeros_like(Y1)
for j in range(num_users):
    Ynew[:,j]=np.ceil(Pr_rat[:,j] + Mu[j])

#np.save("Trained_var",res.x)