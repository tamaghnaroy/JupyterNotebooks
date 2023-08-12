import numpy as np
import matplotlib.pyplot as plt

def covariance(i, H):
    if i==0:
        return 1
    else:
        return ((i-1)**(2*H) - 2*i**(2*H) + (i+1)**(2*H))/2

def fBM_Levinson(m, H, L=1, cumm=1,_seed=None, sigma=1):
    """
    Generate Fractional Brownian Motion Paths
    
    @param m: Length of each path
    @param H: Hurst Exponent
    @param L: Number of such paths
    @param cumm: 1 = cummulative returns
    @param _seed: seed for numpy
    """
    k=np.array(range(m))
    if _seed:
        np.random.seed(_seed)
    scaling = (L/m) ** (2*H)
    
    # -- Covariance
    cov = [covariance(i, H) for i in range(m)]
    
    # -- Initialization of the algorithm
    y = np.random.normal(0,1,m)
    fGn = np.zeros(m)
    v1 = np.array(cov)
    v2 = np.array([0] + cov[1:] + [0])
    k = v2[1]
    aa = np.sqrt(cov[0])
    
    # -- Levinson's algorithm
    for j in range(1,m):
        aa = aa * np.sqrt(1 - k*k)
        v = k * v2[j:m] + v1[j-1:m-1]
        v2[j:m] = v2[j:m] + k * v1[j-1:m-1]
        v1[j:m] = v
        bb = sigma*y[j] / aa
        fGn[j:m] = fGn[j:m] + bb * v1[j:m]
        k = -v2[j+1]/(aa*aa)
    
    # -- scaling and output
    for i in range(m):
        fGn[i] = scaling * fGn[i]
        if cumm and i>0:
            fGn[i] = fGn[i] + fGn[i-1]
    
    return fGn
