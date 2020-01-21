#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Comparing Surface Pressure/XCO2/XCO/XCH4 of a merge file.')
parser.add_argument('-f', '--file', metavar='merge file', help='Merged Scene/L1b/L2 file')
args = parser.parse_args()
file_id = args.file

keys = ['psurf', 'xco2', 'xco', 'xch4']
units = ['(Pa)', '(ppm)', '(ppb)', '(ppb)']

def plot_scat11(truth,retr, lbl = '', sdfilter=5, **kwargs):
    '''
    This function plots a 'hexbinned' 1-to-1 scatter plot with a y=x reference line.
    Points with a difference of greater than 5 standard deviations are filtered out.

    Input:
    truth:          the true quantity to reference.
    retr:           the retrieved quantity to compare.
    lbl:            quantity label (string).
    sdfilter:       number of standard deviations for considering outliers.

    Return:
    A 1-to-1 hexbin plot.
    '''
    ind = np.logical_not(np.isnan(truth))*np.logical_not(np.isnan(retr))
    n = np.sum(ind*1)
    diff = truth-retr
    mu = np.nanmean(diff)
    sd = np.nanstd(diff)
    if not sdfilter is None:
        ind = ind*((np.abs(diff-mu) < sdfilter*sd))
    rho = np.corrcoef(truth[ind], retr[ind])[1,0]
    truthsd = np.std(truth[ind])
    retrsd = np.std(retr[ind])
    plt.hexbin(x=truth[ind], y=retr[ind],mincnt=1, zorder=2, **kwargs)
    cbar = plt.colorbar()
    cbar.set_label('# of soundings')
    plt.xlabel(r'truth %s, $\sigma$ = %.2f' % (lbl,truthsd))
    plt.ylabel(r'retrieved %s, $\sigma$ = %.2f' % (lbl,retrsd))
    plt.title(r'N = %d, $\mu_{diff}$ = %.2f, $\sigma_{diff}$ = %.2f, $\rho$ = %.2f' % (n, mu, sd, rho))

if __name__ == '__main__':
    merged = h5.File(file_id, 'r')
    fig, axs = plt.subplots(2,2, dpi=200, figsize=(16,10))
    axs = axs.ravel()
    for i in range(4):
        plt.sca(axs[i])
        plot_scat11(truth=merged['analysis/flat/' + keys[i] + '_truth'][:], retr=merged['analysis/flat/' + keys[i]][:], lbl=keys[i] + units[i], gridsize=25)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.savefig('four_panel.png', bbox_inches='tight')
    merged.close()
