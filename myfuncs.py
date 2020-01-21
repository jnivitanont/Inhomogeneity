#/home/jnivitanont/software/miniconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def plot_scat11(truth,retr, lbl = '', filtersd=True, sd_threshold=5):
    '''
    This function plots a 'hexbinned' 1-to-1 scatter plot with a y=x reference line.
    Points with a difference of greater than 5 standard deviations are filtered out.
    
    Input:
    truth:          the true quantity to reference.
    retr:           the retrieved quantity to compare.
    lbl:            quantity label (string).
    filtersd:       Filter outliers (boolean).
    sd_threshold:   number of standard deviations for considering outliers.
    
    Return:
    A 1-to-1 hexbin plot.
    '''
    ind = np.logical_not(np.isnan(truth))*np.logical_not(np.isnan(retr))
    n = np.sum(ind*1)
    diff = truth-retr
    mu = np.nanmean(diff)
    sd = np.nanstd(diff)
    if filtersd:
        ind = ind*(np.abs(diff) < sd_threshold*sd)
    plt.subplots(1,1)
    yx = [np.min([truth[ind],retr[ind]]), np.max([truth[ind],retr[ind]])]
    plt.plot(yx,yx, '-k', zorder=1)
    plt.hexbin(x=truth[ind], y=retr[ind], mincnt=1, gridsize=(50,50), zorder=2)
    cbar = plt.colorbar()
    cbar.set_label('Number of soundings')
    plt.xlabel('truth ' + lbl)
    plt.ylabel('retrieved ' + lbl)
    plt.title('N = ' + str(n) + ', mu = ' + str(round(mu, 2)) + ', sd = ' + str(round(sd,2)))

def riemann_sum(x,y):
    '''
    This function calculates the Riemann sum using the midpoint method
    
    Input:
    x: points on x-axis corresponding to y-values.
    y: values of y=f(x).
    
    Return:
    sum
    '''
    dx = np.ediff1d(x)
    mids = (x[:-1] + x[1:]) / 2.
    f_val = np.interp(mids, x, y)
    return dx@f_val

def log_interp(x, xp, fp):
    '''
    This function maps fp into log-space and then does a linear interpolation.
    See documentation for np.interp
    '''
    lfp = np.log(fp)
    linterp = np.interp(x, xp, lfp)
    return np.exp(linterp)

def plot_norm(x, y, log=False, **kwargs):
    '''
    This function normalizes y to have area 1 under the curve.
    '''
    if log:
        plt.semilogy(x, y/riemann_sum(x,y), **kwargs)
    else:
        plt.plot(x, y/riemann_sum(x,y), **kwargs)
