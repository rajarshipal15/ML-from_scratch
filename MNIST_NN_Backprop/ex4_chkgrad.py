#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:08:30 2021

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

n1=3 # number of units in input layer
n2=4 # number of units in hidden layer
n3=3 #number of units in output layer

m=10 #number of training examples
K=n3
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigGrad(z):
    return sigmoid(z)*(1.0-sigmoid(z))
tmp1=scipy.io.loadmat(file)
X,y=tmp1['X'],tmp1['y'] 

X,y=X[0:m,0:3],y[0:m]
X=np.concatenate((np.reshape(np.ones(m),(m,1)),X),axis=1)
y=y.reshape(m,)
yp=np.zeros((m,K))
for i in range(m):
    temp=np.zeros(K)
    temp[y[i] % 3]=1.0
    yp[i,:]=temp
n=n3*(n2+1) + n2*(n1+1) 
def Cost(theta): #vectorized cost
#    n=n3*(n2+1) + n2*(n1+1) 
    Theta1=theta[0:n2*(n1+1)].reshape((n2,n1+1))
    Theta2=theta[n2*(n1+1):n ].reshape((n3,n2+1))
    xp=np.zeros((m,K))
    for i in range(m):
        tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[i,:])))
        xp[i,:]=tmpc
    cost=(1.0/m)*np.sum(-yp*np.log(xp)-(1.0-yp)*np.log(1.0-xp))
    return cost 
#tmpc= sigmoid(Theta2  @ np.append(1,sigmoid(Theta1 @ X[i,:])))
#xp[i,:]=tmpc
#thetap,thetam=np.zeros(n),np.zeros(n)
eps=0.01
gradtheta=np.zeros(n)
#thetaini=np.random.uniform(-1,1,n)
thetaini=np.zeros(n)
for i in range(n):
#    thetap,thetam=np.zeros(n),np.zeros(n)
    thetap=np.asarray([k for k in thetaini])
    thetam=np.asarray([k for k in thetaini])
    thetap[i],thetam[i]=thetap[i]+ eps, thetam[i]-eps     
    gradtheta[i]=(Cost(thetap)-Cost(thetam))/(2.0*eps)

#Gradient obtained by backprop
def Grad(theta):
    Theta1=theta[0:n2*(n1+1)].reshape((n2,n1+1)) #4x4
    Theta2=theta[n2*(n1+1):n ].reshape((n3,n2+1)) #3x5    
    DEL1,DEL2 = np.zeros_like(Theta1), np.zeros_like(Theta2)
    for i in range(m):
        a1=X[i,:]   # 4x
        z2=Theta1 @ a1 #4x1
        a2=sigmoid(z2) #4x1
        a2=np.append(1,a2) #5x1
        z3=Theta2 @ a2 # 3x1           
        a3=sigmoid(z3) #Output layer 3x1
        delta3=np.zeros_like(a3) #3x1
    #    for K in range(10):
    #        yp=(y==K).astype(float)
        delta3=a3-yp[i,:] #3x1
        delta2= ((Theta2.T) @ delta3) * (np.append(1,sigGrad(z2)))#5x1
        delta2=np.delete(delta2,0) #4x1
        DEL1= DEL1 + (delta2.reshape(4,1)) @ (a1.reshape(1,4))   
        DEL2 = DEL2 + (delta3.reshape(3,1)) @ (a2.reshape(1,5))
#    GradTheta1, GradTheta2=DEL1/m,DEL2/m
    return np.concatenate(((DEL1/m).flatten(),(DEL2/m).flatten()))

temp=np.max(abs(Grad(thetaini)-gradtheta))
    
 