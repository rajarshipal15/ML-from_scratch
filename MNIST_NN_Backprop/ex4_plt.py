#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:37:29 2021

@author: rajarshi
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from PIL import Image
import scipy.io
# Creates a random image 100*100 pixels
#mat = np.random.random((100,100))

# Creates PIL image
#img = Image.fromarray(mat, 'L')
#img.show()
file='/media/rajarshi/My Passport/Home/codes/Ng-ml/Ex4/ex4data1.mat'

tmp = scipy.io.loadmat(file)
X,y=tmp['X'],tmp['y']
y[y > 9.0]=0.0

# tmp2=(X[4980,:].reshape(20,20)).T
# #img = Image.fromarray(tmp2, 'L')
# #img.show()

# fig = plt.figure
# plt.imshow(tmp2, cmap='gray_r')
# plt.show()

fig, axs = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        tmp2=(X[random.randint(0,4999),:].reshape(20,20)).T
        axs[i,j].imshow(tmp2, cmap='gray_r') 
        axs[i,j].axis('off') 







