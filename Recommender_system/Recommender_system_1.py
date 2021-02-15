#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:21:34 2021

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
lamb=1.5
Cost=0.5*np.sum(tmp*tmp) + 0.5*lamb*(np.sum(X*X) + np.sum(Theta*Theta))
#Cost obtained is 22.2246 for lamb=0.0
# Cost obtained for lamb=1.5 is 31.34
print("Cost obtained for the regulation parameter lambda =%f is %f"%(lamb,Cost) )


