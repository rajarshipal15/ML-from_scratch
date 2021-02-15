#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:18:20 2021

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

res=np.load("Trained_var.npy")


X=res[0:num_movies*num_features].reshape((num_movies,num_features)) #4x4
Theta=res[num_movies*num_features:n ].reshape((num_users,num_features))

Pr_rat= (X @ Theta.T)
Ynew=np.zeros_like(Y1)
for j in range(num_users):
    Ynew[:,j]=np.ceil(Pr_rat[:,j] + Mu[j])

movie_names=open('movie_ids.txt',encoding = "ISO-8859-1")
lines=movie_names.readlines()

max1,max2=np.max(Y[:,0]),np.max(Ynew[:,0])
idx1=np.where(Y[:,0]==max1)[0]
idx2=np.where(Ynew[:,0]==max2)[0]
idx0=np.where(R[:,0]==1)[0]
#idx3=np.setdiff1d(idx2,idx1)
idx3=np.setdiff1d(idx2,idx0)
print("The top recommendations for the first user, for movies not rated by him/her in the original dataset are")
for k in idx3:
    print(lines[k])
    
        



