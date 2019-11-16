#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style("white")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared)
from sklearn import preprocessing
from sklearn import metrics


# In[2]:


#y is the CO2 and x is the year
yTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,1]
xTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,2]
yActual = pd.read_csv('./Data/CO2data.csv').iloc[201:301,1]
xActual = pd.read_csv('./Data/CO2data.csv').iloc[201:301,2]

p = 200  # Number of inputs
t = np.linspace(-np.pi, np.pi, 100)

# Preallocate a matrix (make it a matrix of ones to account for beta_0):
XS = np.ones((100, p)) 
for ii in range(1, p):
    XS[:, ii] = np.sin((ii+1)*t)
beta = np.random.randn(p)
ff = np.dot(XS, beta)
plt.plot(t,ff,'o')
plt.plot(t,XS)

plt.xlim([-np.pi, np.pi])
plt.xlabel('t')
plt.ylabel('xs')

#Orginal data for first 200 months(Blue) last 301 months (green)

#SM Kernel (black)

#matern (cyan)

#Squared exponential (dashed red)

#Rational Quadratic (magenta)

#Periodic kernel (orange)

#2 stdev about the mean 95% of predictive mass (light grey)


# In[ ]:


def SM_kernel(x, y, param)


# In[ ]:


def matern(x, y,param)


# In[ ]:


def squared_exponential(x, y,param)


# In[ ]:


def rational_quadratic(x, y,param)


# In[ ]:


def periodic_kernel(x, y, param)


# In[ ]:


#log spectral densities of the leanred SM (black)


#squared exponential kernals (red)

