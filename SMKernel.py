import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style("white")
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, 
                                              ExpSineSquared, WhiteKernel, ConstantKernel as C)
from sklearn import preprocessing
from sklearn import metrics


def SM_kernel(xTrain, yTrain, xTest):
    k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
    k2 = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
    k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
    k4 = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
   
    smkernel = k1+k2+k3+k4
    gp = GaussianProcessRegressor(kernel=smkernel, alpha=0,
                              optimizer=None, normalize_y=True)
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    X_ = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    return X_, y_pred, y_std
    

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
    
def rational_quadratic(xTrain, yTrain, xTest):
    noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)
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


def periodic_kernel(xTrain, yTrain, xTest):
    noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)
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

def squared_exponential(xTrain, yTrain, xTest):
    k = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
    noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
    rbf = k  + noise
    gp = GaussianProcessRegressor(kernel=rbf, alpha=0,
                                  optimizer=None, normalize_y=True)
    
    xTrain= np.asarray(xTrain).reshape(-1, 1)
    yTrain= np.asarray(yTrain).reshape(-1, 1)
    xTest = np.asarray(xTest).reshape(-1, 1)
    gp.fit(xTrain, yTrain)
    
    X = np.linspace(xTest.min(), xTest.max(), 200)[:, np.newaxis]
    y_pred, y_std = gp.predict(X, return_std=True)
    
    
    return X , y_pred


  
  
  
#start of combining the kernels onto a plot and loading the data
#y is the CO2 and x is the year
xTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,4]
yTrain = pd.read_csv('./Data/CO2data.csv').iloc[1:200,1]
xTest = pd.read_csv('./Data/CO2data.csv').iloc[201:501,4]
yTest = pd.read_csv('./Data/CO2data.csv').iloc[201:501,1]

x = pd.read_csv('./Data/CO2data.csv').iloc[1:501,4]
y = pd.read_csv('./Data/CO2data.csv').iloc[1:501,1]

#Orginal data for first 200 months(Blue) last 301 months (green)
plt.figure(1)
plt.plot(xTest,yTest, "green", label='Original data')
plt.plot(xTrain, yTrain, "blue", label='Training Data')

#SM Kernel (black)
X, y_pred,y_std = SM_kernel(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"black", label='SM Kernel')
y1=np.ravel(y_pred)-np.ravel(y_std);
y2=np.ravel(y_pred)+np.ravel(y_std);


#2 stdev about the mean 95% of predictive mass (light grey)

plt.fill_between(X[:, 0], y1, y2, alpha=0.5, color='lightgrey', label='STD 95%')

#matern (cyan)
X, y_pred = matern(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"cyan", label='Matern')


#Squared exponential (dashed red)
X, y_pred = squared_exponential(xTrain, yTrain, xTest)
plt.plot(X, y_pred,'r--', label='RBF')


#Rational Quadratic (magenta)
X, y_pred = rational_quadratic(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"magenta", label='Rational Quadratic')


#Periodic kernel (orange)
X, y_pred = periodic_kernel(xTrain, yTrain, xTest)
plt.plot(X, y_pred,"orange", label='periodic Kernel')


plt.suptitle('Mauna Loa CO2 Concentration')
plt.xlabel('Year')
plt.ylabel('CO2 Concentration(ppm)')
plt.legend()
plt.savefig('./Figures/replicateFig.jpg')
plt.show()



def kernel(hypcov, x=None, z=None, diag=False):
    
    n, D = x.shape
    hypcov = np.array(hypcov).flatten()
    Q = hypcov.size/(1+2*D)
    w = np.exp(hypcov[0:Q])
    m = np.exp(hypcov[Q+np.arange(0,Q*D)]).reshape(D,Q)
    v = np.exp(2*hypcov[Q+Q*D+np.arange(0,Q*D)]).reshape(D,Q)
    d2list = []
    
    if diag:
        d2list = [np.zeros((n,1))]*D
    else:
        if x is z:
            d2list = [np.zeros((n,n))]*D
            for j in np.arange(0,D):
                xslice = np.atleast_2d(x[:,j])
                d2list[j] = sp.spatial.distance.cdist(xslice, xslice, 'sqeuclidean')
        else:
            d2list = [np.zeros((n,z.shape[0]))]*D
            for j in np.arange(0,D):
                xslice = np.atleast_2d(x[:,j])
                zslice = np.atleast_2d(z[:,j])
                d2list[j] = sp.spatial.distance.cdist(xslice, zslice, 'sqeuclidean')

    # Define kernel functions
    k = lambda d2v, dm: np.multiply(np.exp(-2*np.pi**2 * d2v),
                                    np.cos(2*np.pi * dm))
    # Calculate correlation matrix
    K = 0
    # Need the sqrt
    dlist = [ np.sqrt(dim) for dim in d2list ]
    # Now construct the kernel
    for q in range(0,Q):
        C = w[q]**2
        for j,(d,d2) in enumerate(zip(dlist, d2list)):
            C = np.dot(C, k(np.dot(d2, v[j,q]), 
                            np.dot(d, m[j,q])))
        K = K + C
    return K


def initSMParamsFourier(Q, x, y, sn, samplingFreq, nPeaks, relMaxOrder=2):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, D = x.shape
    w = np.zeros(Q)
    m = np.zeros((D,Q))
    s = np.zeros((D,Q))
    w[:] = np.std(y) / Q
    hypinit = {
        'cov': np.zeros(Q+2*D*Q),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }

    # Assign hyperparam weights
    hypinit['cov'][0:Q] = np.log(w)

    # Assign hyperparam frequencies (mu's)
    signal = np.array(y.ravel()).ravel()  # Make into 1D array
    n = x.shape[0]
    k = np.arange(n)
    ts = n/samplingFreq
    frqx = k/float(ts)
    rng = range(n/2)
    frqx = frqx[rng]
    
    
    
    
    
    frqy = np.fft.fft(signal)/n
    print(frqy)
    frqy = abs(frqy[range(n/2)])
    print(frqy)
    # Find the peaks in the frequency spectrum
    peakIdx = np.array([frqy])
    print(peakIdx)
    sortedIdx = frqy[peakIdx].argsort()[::-1][:nPeaks]
    sortedPeakIdx = peakIdx[sortedIdx]
    hypinit['cov'][Q + np.arange(0,Q*D)] = np.log(frqx[sortedPeakIdx])

    # Assign hyperparam length scales (sigma's)
    for i in range(0,D):
        xslice = np.atleast_2d(x[:,i])
        d2 = spat.distance.cdist(xslice, xslice, 'sqeuclidean')
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        maxshift = np.max(np.max(np.sqrt(d2)))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))
    hypinit['cov'][Q + Q*D + np.arange(0,Q*D)] = np.log(s[:])
    
    return hypinit


def initSMParams(Q, x, y, sn):
    """
    Initialize hyperparameters for the spectral-mixture kernel. Weights are
    all set to be uniformly distributed, means are given by a random sample
    from a uniform distribution scaled by the Nyquist frequency, and variances 
    are given by a random sample from a uniform distribution scaled by the max 
    distance.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, D = x.shape
    w = np.zeros(Q)
    m = np.zeros((D,Q))
    s = np.zeros((D,Q))
    w[:] = np.std(y) / Q
    hypinit = {
        'cov': np.zeros(Q+2*D*Q),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }

    for i in range(0,D):
        # Calculate distances
        xslice = np.atleast_2d(x[:,i])
        d2 = spat.distance.cdist(xslice, xslice, 'sqeuclidean')
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        minshift = np.min(np.min(np.sqrt(d2)))
        nyquist = 0.5/minshift
        m[i,:] = nyquist*np.random.ranf((1,Q))
        maxshift = np.max(np.max(np.sqrt(d2)))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))

    hypinit['cov'][0:Q] = np.log(w)
    hypinit['cov'][Q + np.arange(0,Q*D)] = np.log(m[:]).T
    hypinit['cov'][Q + Q*D + np.arange(0,Q*D)] = np.log(s[:]).T
    return hypinit


def initBoundedParams(bounds, sn=[]):
    hypinit = {
        'cov': np.zeros(len(bounds)),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }
    # Sample from a uniform distribution
    for idx, pair in enumerate(bounds):
        # Randomize only if bounds are specified
        if isinstance(pair, collections.Iterable):
            hypinit['cov'][idx] = np.random.uniform(pair[0], pair[1])
        # If no bounds, then keep default value always
        else:
            hypinit['cov'][idx] = pair
    return hypinit

