#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:35:51 2021

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
import sklearn as skt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
file='ex7data2.mat'
K=3
tmp1=scipy.io.loadmat(file)
X=tmp1['X']
m,n=len(X[:,0]), len(X[0,:])
#centroids_ini=np.random.uniform(np.min(X),np.max(X),(K,n))
#index = np.random.choice(m, K, replace=False) 
#centroids_ini=X[index,:]
centroids_ini=np.load('centroids_ini.npy')
#centroids_ini=np.load('centroids_ini.npy')
def plot_sorted(x,y,s):
    import itertools
    xs, ys = zip(*sorted(zip(x, y)))
    plt.plot(xs,ys,s)

def findClosestCentroids(X, centroids):
    tmp=np.zeros(m)
    for i in range(m):
        tmp1=[norm(X[i,:]-centroids[j,:]) for j in range(K)]
        tmp[i]=tmp1.index(np.min(tmp1)) 
    return tmp
    
def computeCentroids(X,centroids,position):
    for i in range(K):
        tmp1=np.where(position==i)[0]
        if (len(tmp1) != 0): 
            tmpsum=np.zeros(n)
            for j in tmp1:
                tmpsum += X[j,:] 
            tmpsum=tmpsum/len(tmp1) 
            centroids[i,:]=tmpsum
    return centroids
    # for i in tmp:
    #     tmp1=np.where(tmp==i)[0]
    #     tmpsum=np.zeros(n)
    #     for j in tmp1:
    #         tmpsum = tmpsum + X[j,:]
    #     tmpsum = tmpsum/len(tmp1)
#centroids=centroids_ini
plot_sorted(X[:,0],X[:,1],'o')
def iterations_kmeans(X,centroids,num):
    for iterations in range(num):
        idk=findClosestCentroids(X, centroids)
        centroids=computeCentroids(X, centroids, idk)
    return centroids
num=10
# centroids_final=iterations_kmeans(X,centroids_ini,num)
plt.plot(centroids_ini[:,0],centroids_ini[:,1],'^',markersize=20,label="Initial position")    
x1,y1,x2,y2,x3,y3=np.array(centroids_ini[:,0][0]),np.array(centroids_ini[:,1][0]),np.array(centroids_ini[:,0][1]),np.array(centroids_ini[:,1][1]),np.array(centroids_ini[:,0][2]),np.array(centroids_ini[:,1][2])

for i in range(1,num):
    centroids_final=iterations_kmeans(X,centroids_ini,i)
    x1,y1=np.append(x1,centroids_final[:,0][0]), np.append(y1,centroids_final[:,1][0])
    x2,y2=np.append(x2,centroids_final[:,0][1]), np.append(y2,centroids_final[:,1][1])
    x3,y3=np.append(x3,centroids_final[:,0][2]), np.append(y3,centroids_final[:,1][2])
#    plt.plot(centroids_final[:,0][0],centroids_final[:,1][0],'x',markersize=15,color='black')
#    plt.plot(centroids_final[:,0][1],centroids_final[:,1][1],'o',markersize=15,color='red')
#    plt.plot(centroids_final[:,0][2],centroids_final[:,1][2],'s',markersize=15,color='green')
plt.plot(x1,y1,'x--',color='black')    
plt.plot(x2,y2,'x--',color='black')  
plt.plot(x3,y3,'x--',color='black')  
plt.plot(centroids_final[:,0],centroids_final[:,1],'^',markersize=20,color='green',alpha=0.7,label='Final position')
plt.legend(loc=0)