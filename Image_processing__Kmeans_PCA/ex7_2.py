#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:22:40 2021

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

sys.path.append('/media/rajarshi/My Passport/Home/codes/Ng-ml/Ex7')
image = image.imread('/media/rajarshi/My Passport/Home/codes/Ng-ml/Ex7/bird_small.png')

X=image.reshape(128**2,3)
m,n=len(X[:,0]), len(X[0,:])
# K is number of centroids
K=16
num=30
index = np.random.choice(m, K, replace=False) 
centroids_ini=X[index,:]

#from ex7_1 import findClosestCentroids, computeCentroids, iterations_kmeans


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

def iterations_kmeans(X,centroids,num):
    for iterations in range(num):
        idk=findClosestCentroids(X, centroids)
        centroids=computeCentroids(X, centroids, idk)
    return centroids

centroids_final=iterations_kmeans(X,centroids_ini,num)
#np.save('image_kmeans',centroids_final)

position=findClosestCentroids(X, centroids_final)
X1=np.zeros_like(X)
for i in range(m):
    X1[i,:]=centroids_final[int(position[i]),:]
X2=X1.reshape(128,128,3)
#fig, (ax1, ax2) = plt.subplots(1, 2)
#fig.suptitle('Horizontally stacked subplots')
#ax1.imshow(X)
#ax2.imshow(X2)

plt.imshow(X)
plt.imshow(X2)



