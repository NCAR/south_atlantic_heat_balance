# Thanks to Dr. Paulo Polito for providing the code that served as the basis for creating this function 
'''
Little test of Monte Carlo using Pearson correlation
'''
# Imports

import datetime as da
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import detrend
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.io import savemat
from scipy import signal
import netCDF4
import xarray as xr

# Function 1
def dec_scale(x):
    ''' Return the decorrelation scale of x in point units 
    (yes, you should multiply by the data resolution)
    '''
    x = detrend(x)
    um = np.ones(x.shape)
    ac = np.correlate(x, x, mode='full')
    uc = np.correlate(um, um, mode='full')
    # remove bias, i.e.: correct for the linear decay
    ac = ac/uc
    l0 = int((ac.shape[0]-1)/2)
    ac = ac[l0:int(1.5*l0)]
    ac = ac/ac[0]
    n = ac.shape[0]
    to = np.arange(n)
    fi = interp1d(to, ac)
    tf = np.linspace(to[0], to[-1], 10*n)
    acf = fi(tf)
    idec = np.where(acf<=np.exp(-1))[0][0]
    return tf[idec]

# Function 2

def MC_corr(ds1,ds2,nrd,mlags):
    ''' With this function you will correlate the data sets 
    ds1 and ds2 and use nrd random serial numbers and mlags represents the maximum lag. 
    '''
    ds1=signal.detrend(ds1) # Squeeze out extra dimension and remove linear trend 

    ds2=signal.detrend(ds2) # Squeeze out extra dimension and remove linear trend

    m=nrd
    
    matrix = plt.xcorr(ds1[:], ds1[:], maxlags=mlags, normed=True) # Maximum Lag

    lags=(matrix*1)[0]; cc=(matrix*1)[1] # The first output of the function is lag and the second is correlation

    plt.close('all')

    return cc

    # Determining the lag for maximum correlation (positive or negative)

#    if cc[np.where(cc==max(cc))] > np.abs(cc[np.where(cc==min(cc))]): # cc is the correlation index and ci is the maximum correlation index
#        ci=cc[np.where(cc==max(cc))]; lag=lags[np.where(cc==max(cc))]
#    else:
#        ci=cc[np.where(cc==min(cc))]; lag=lags[np.where(cc==min(cc))]

#    del cc, lags

#    idec1 = dec_scale(ds1[:])
#    idec2 = dec_scale(ds2[:])
#    n = int((ds1[:].shape[0]/idec1 + ds2[:].shape[0]/idec2) / 2); del idec1, idec2
#    fake1 = np.random.randn(m, n)
#    fake2 = np.random.randn(m, n)
#    del n
#    mc_ci = np.empty(m)
#    for i in np.arange(m):
#        matrix = plt.xcorr(fake1[i, :], fake2[i, :], usevlines=True, maxlags=1, normed=True, lw=2);
#        cc=(matrix*1)[1]; del matrix
#    return cc
#        if cc[np.where(cc==max(cc))] > np.abs(cc[np.where(cc==min(cc))]):
#            mc_ci[i]=cc[np.where(cc==max(cc))];
#        else:
#            mc_ci[i]=cc[np.where(cc==min(cc))];
#    del cc, fake1, fake2

#    p10 = 0.5*np.sum(np.abs(mc_ci) > np.abs(ci[0]))/m

#    plt.close('all')

#    return ci, p10, lag # correlation index, pvalue, lag

