
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style("white")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, 
                                              ExpSineSquared, WhiteKernel)
from sklearn import preprocessing
from sklearn import metrics

def squared_exponential(xTrain, yTrain, xTest):
    
    k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
    noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
    kernel_gpml = noise
    gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                                  optimizer=None, normalize_y=True)
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    
    X_ = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    
    
    return X_ , y_pred


# In[194]:


#y is the CO2 and x is the year
xTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,4]
yTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,1]
xTest = pd.read_csv('./Data/CO2data.csv').iloc[201:501,4]
yTest = pd.read_csv('./Data/CO2data.csv').iloc[201:501,1]

x = pd.read_csv('./Data/CO2data.csv').iloc[1:501,4]
y = pd.read_csv('./Data/CO2data.csv').iloc[1:501,1]

#Orginal data for first 200 months(Blue) last 301 months (green)
plt.figure(1)
plt.plot(xTest,yTest, "green")
plt.plot(xTrain, yTrain, "blue")

#SM Kernel (black)



#matern (cyan)
X, y_pred = matern(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"cyan")



#Squared exponential (dashed red)
X, y_pred = squared_exponential(xTrain, yTrain, xTest)
plt.plot(X, y_pred,'r--')


#Rational Quadratic (magenta)
X, y_pred = rational_quadratic(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"magenta")


#Periodic kernel (orange)
X, y_pred = periodic_kernel(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"orange")


#2 stdev about the mean 95% of predictive mass (light grey)

plt.suptitle('Mauna Loa CO2 Concentration')
plt.xlabel('Year')
plt.ylabel('CO2 Concentration(ppm)')
plt.show()


# In[86]:


def SM_kernel():
    


# In[191]:


def matern(xTrain, yTrain, xTest):
    matern = Matern(length_scale=0.25)
    gp = GaussianProcessRegressor(kernel=matern, alpha=0,
                                  optimizer=None, normalize_y=True)
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    
    X_ = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    return X_, y_pred
    


# In[192]:


def rational_quadratic(xTrain, yTrain, xTest):
    rational = 0.66**2* RationalQuadratic(length_scale=1.2, alpha=0.78) + noise
    gp = GaussianProcessRegressor(kernel= rational, alpha=0,
                                  optimizer=None, normalize_y=True)
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    
    X_ = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    return X_, y_pred


# In[193]:


def periodic_kernel(xTrain, yTrain, xTest):
    periodic = 2.4**2 * ExpSineSquared(length_scale=1.3, periodicity=1.0) +noise
    gp = GaussianProcessRegressor(kernel=periodic, alpha=0,
                                  optimizer=None, normalize_y=True)
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    
    X_ = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    return X_, y_pred


# In[ ]:





# In[ ]:


def InitSMhypers (q,x,y,flag)
    # create hypers
    w = np.zeros(1,Q);
    m = np.zeros(D,Q);
    s = np.zeros(D,Q);

    # create initialisation vector of all hypers
    hypinit = zeros(Q+2*D*Q,1);

    emp_spect = abs(fft(y)).^2/N;
    log_emp_spect = abs(log(emp_spect));

    M = floor(N/2);

    freq = [[0:M],[-M+1:1:-1]]'/N; 

    freq = freq(1:M+1);
    emp_spect = emp_spect(1:M+1);
    log_emp_spect = log_emp_spect(1:M+1);

