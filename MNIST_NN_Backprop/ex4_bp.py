#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:44:23 2021

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
import time

start_time = time.time()
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
K=10
Y=np.zeros((m,K))
for i in range(m):
    temp=np.zeros(K)
    temp[y[i]]=1.0
    Y[i,:]=temp

eps=0.12
Theta1ini=np.random.uniform(-eps,eps,(25,401))
Theta2ini=np.random.uniform(-eps,eps,(10,26))
DEL1,DEL2 = np.zeros_like(Theta1ini), np.zeros_like(Theta2ini)
n1,n2,n3=400,25,10
n=n3*(n2+1) + n2*(n1+1)
lamb=1.0
def Cost(theta): #vectorized cost
#    n=n3*(n2+1) + n2*(n1+1) 
    Theta1=theta[0:n2*(n1+1)].reshape((n2,n1+1))
    Theta2=theta[n2*(n1+1):n ].reshape((n3,n2+1))
    xp=np.zeros((m,K))
    for i in range(m):
        tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[i,:])))
        xp[i,:]=tmpc
    cost=(1.0/m)*np.sum(-Y*np.log(xp)-(1.0-Y)*np.log(1.0-xp)) + (lamb/(2.0 *m))*(np.sum(Theta1[:,1:]*Theta1[:,1:])+ np.sum(Theta2[:,1:]*Theta2[:,1:]))
    return cost 

def Grad(theta):
    Theta1=theta[0:n2*(n1+1)].reshape((n2,n1+1)) #4x4
    Theta2=theta[n2*(n1+1):n ].reshape((n3,n2+1)) #3x5    
    DEL1,DEL2 = np.zeros_like(Theta1), np.zeros_like(Theta2)
    for i in range(m):
        a1=X[i,:]
        z2=Theta1 @ a1
        a2=sigmoid(z2)
        a2=np.append(1,a2)
        z3=Theta2 @ a2             
        a3=sigmoid(z3) #Output layer
        delta3=np.zeros_like(a3)
        for K in range(10):
            yp=float(y[i]==K)
            delta3[K]=a3[K]-yp
        delta2= ((Theta2.T) @ delta3) * (np.append(1,sigGrad(z2)))
        delta2=np.delete(delta2,0)
        DEL1= DEL1 + (delta2.reshape(25,1)) @ (a1.reshape(1,401))   
        DEL2 = DEL2 + (delta3.reshape(10,1)) @ (a2.reshape(1,26))
        tmpa= np.append(np.zeros(n2).reshape(n2,1),np.delete(Theta1,0,1),axis=1) 
        tmpa1=np.append(np.zeros(n3).reshape(n3,1),np.delete(Theta2,0,1),axis=1)
    GradTheta1= (DEL1 + lamb*tmpa)/m
    GradTheta2=(DEL2 + lamb*tmpa1)/m
    return f(GradTheta1,GradTheta2)

thetaini=f(Theta1ini,Theta2ini)
res = opt.minimize(Cost, thetaini, method='BFGS', jac=Grad,options={'disp': True,'maxiter': 400})
Theta1f=res.x[0:n2*(n1+1)].reshape((n2,n1+1))
Theta2f=res.x[n2*(n1+1):n ].reshape((n3,n2+1)) 
#np.save('Theta1f-400it',Theta1f)
#np.save('Theta2f-400it',Theta2f)
end_time=time.time()
time_taken=end_time - start_time 
