#!/usr/bin/env python3

import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
from descartes.patch import PolygonPatch
import seaborn as sns
import shapely.geometry as sgeom
import pandas as pd

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.bbox'] = 'tight'
cmap_hotr = mpl.cm.hot_r
cmap_hotr.set_bad('green', 1.)
cmap_seis = mpl.cm.seismic
cmap_seis.set_bad('yellow', 1.)

ils_cv_keys = [
    'ils_shape_error_ch4',
    'ils_shape_error_o2',
    'ils_shape_error_strong_co2',
    'ils_shape_error_weak_co2',
    'subslit_albedo_cv_o2',
    'subslit_albedo_cv_strong_co2',
    'subslit_albedo_cv_weak_co2',
    'subslit_albedo_slope_o2',
    'subslit_albedo_slope_strong_co2',
    'subslit_albedo_slope_weak_co2']
gas_keys=[
    'dP',
    'xco2_error',
    'xco_error',
    'xch4_error']


def plot_hex1to1(x,y, xlbl='', ylbl='', sdfilter=5, **kwargs):
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
    ind = np.logical_not(np.isnan(x))*np.logical_not(np.isnan(y))
    n = np.sum(ind*1)
    diff = x-y
    mu = np.nanmean(diff)
    sd = np.nanstd(diff)
    if not sdfilter is None:
        ind = ind*((np.abs(diff-mu) <  sdfilter*sd))
    rho = np.corrcoef(x[ind], y[ind])[1,0]
    xsd = np.std(x[ind])
    ysd = np.std(y[ind])
    plt.hexbin(x=x[ind], y=y[ind],mincnt=1, zorder=2, **kwargs)
    cbar = plt.colorbar()
    cbar.set_label('# of soundings')
    plt.xlabel(r'%s, $\sigma$ = %.2f' % (xlbl,xsd))
    plt.ylabel(r'%s, $\sigma$ = %.2f' % (ylbl,ysd))
    plt.title(r'N = %d, $\mu_{diff}$ = %.2f, $\sigma_{diff}$ = %.2f, $\rho$ = %.2f' % (n, mu, sd, rho))


def plot_four_panel(merge_file, title=''):
    '''

    '''
    var = ['psurf', 'xco2', 'xco', 'xch4']
    unit = ['(Pa)', '(ppm)', '(ppb)', '(ppb)']
    fig, axs = plt.subplots(2,2, figsize=(16,10))
    axs = axs.ravel()
    for i in range(4):
        plt.sca(axs[i])
        plot_hex1to1(merge_file['analysis//' + var[i] + '_truth'][:],
                        merge_file['analysis//'+var[i]][:],
                        'Truth '+var[i]+unit[i],
                        'Retrieved '+var[i]+unit[i],
                        gridsize=25)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    fig.suptitle(title, y=.95,fontsize=20)

def plot_four_panel_kde(merge_file, comp_file, centered=True, sdfilter = 4, title=''):
    '''

    '''
    var = ['psurf_error', 'xco2_error', 'xco_error', 'xch4_error']
    unit = ['(Pa)', '(ppm)', '(ppb)', '(ppb)']
    fig, axs = plt.subplots(2,2, figsize=(16,10))
    axs = axs.ravel()
    for i in range(4):
        merge_var = merge_file['analysis/'][var[i]][np.logical_not(np.isnan(merge_file['analysis/'][var[i]][:]))]
        if not sdfilter is None:    
            mu_merge = np.nanmedian(merge_var)
            sd_merge = np.nanstd(merge_var)
            ind = (np.abs(merge_var-mu) <  sdfilter*sd)
            merge_var = merge_var[ind]
        N_merge = np.size(merge_var)
        mu_merge = np.nanmedian(merge_var)
        sd_merge = np.nanstd(merge_var)
        if not centered:
            z_merge = merge_var
        else:
            z_merge = (merge_var-mu_merge)
        sns.kdeplot(z_merge, shade=True, ax=axs[i], label=r'w/ sh, N = %d, $\mu$ = %.4e, $\sigma$ = %.4e'%( N_merge, mu_merge, sd_merge))
        comp_var = comp_file['analysis/'][var[i]][np.logical_not(np.isnan(comp_file['analysis/'][var[i]][:]))]
        if not sdfilter is None:    
            mu_comp = np.nanmedian(comp_var)
            sd_comp = np.nanstd(comp_var)
            ind = (np.abs(comp_var-mu) <  sdfilter*sd)
            comp_var = comp_var[ind]
        N_comp = np.size(comp_var)
        mu_comp = np.nanmedian(comp_var)
        sd_comp = np.nanstd(comp_var)
        if not centered:
            z_comp = comp_var
        else:
            z_comp = (comp_var-mu_comp)
        sns.kdeplot(z_comp, shade=True, ax=axs[i], label=r'w/o sh, N = %d, $\mu$ = %.4e, $\sigma$ = %.4e'%( N_comp, mu_comp, sd_comp))
        max_sd = np.max([sd_merge, sd_comp])
        axs[i].set_xlim([mu_merge-4*max_sd, mu_merge+4*max_sd])
        axs[i].legend()
        axs[i].set_xlabel(unit[i])
        if not centered:
            axs[i].set_title(var[i])
        else:
            axs[i].set_title(var[i] + ' (centered)')
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    fig.suptitle(title, y=.95,fontsize=20)

def plot_four_panel_spatial(merge_file, lon_grid, lat_grid, site,  title='', apply_ppf=False):
    '''

    '''
    var = ['psurf_error', 'xco2_error', 'xco_error', 'xch4_error']
    unit = ['(Pa)', '(ppm)', '(ppb)', '(ppb)']
    vmaxs = [3000, 5, 5, 15]
    if apply_ppf:
        ind_ppf = post_processing_filter0(merge_file)
    fig, axs = plt.subplots(2,2, figsize=(16,10))
    axs = axs.ravel()
    for i in range(4):
        merge_var = np.abs(merge_file['analysis/'][var[i]][:])
        if apply_ppf:
            merge_var[~ind_ppf] = np.nan
        N_var = sum(~np.isnan(merge_var))
        var_std = np.nanstd(merge_var)
        var_med = np.nanmedian(merge_var)
        mesh = axs[i].pcolormesh(lon_grid, lat_grid, merge_var.reshape(lon_grid.shape),
                               vmin=0,
                               vmax=vmaxs[i], cmap=cmap_hotr)
        axs[i].add_patch(PolygonPatch(polygon=sgeom.Point(site).buffer(.015), color='purple',alpha=1, zorder=6))
        axs[i].set_title(r'%s, $\mu$=%.2f, $\sigma$=%.2f, N=%d' % (var[i], var_med, var_std, N_var) )
        axs[i].set_ylabel('lat')
        axs[i].set_xlabel('lon')
        cbar = plt.colorbar(mesh, ax=axs[i])
        cbar.set_label('absolute error ' + unit[i], rotation=270, labelpad=10)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    fig.suptitle(title + ' (filtered)' if apply_ppf else title, y=.95,fontsize=20)



def plot_medians(merge_file, comp_file, xvars, yvars, mergelbl='', complbl='', plot_directory='./', apply_ppf=False):
    '''

    '''
    if apply_ppf:
        ind_ppf_merge = post_processing_filter0(merge_file)
        ind_ppf_comp = post_processing_filter0(comp_file)


    for xvar in xvars:
        if apply_ppf:
            merge_xvar = merge_file['analysis/'][xvar][ind_ppf_merge]
            comp_xvar = comp_file['analysis/'][xvar][ind_ppf_comp]
        else:
            merge_xvar = merge_file['analysis/'][xvar][:]
            comp_xvar = comp_file['analysis/'][xvar][:]
        xvar_std = np.nanstd(merge_xvar)
        xvar_med = np.nanmedian(merge_xvar)
        xvar_plotmax = xvar_med+xvar_std*5
        xvar_plotmin = np.nanmin(merge_xvar)
        cv_bins = np.linspace(xvar_plotmin, xvar_plotmax, 11)
        binned_merge = np.digitize(merge_xvar, bins=cv_bins)
        binned_comp = np.digitize(comp_xvar, bins=cv_bins)
        for yvar in yvars:
            if apply_ppf:
                merge_yvar = merge_file['analysis/'][yvar][ind_ppf_merge]
                comp_yvar = comp_file['analysis/'][yvar][ind_ppf_comp]
            else:
                merge_yvar = merge_file['analysis/'][yvar][:]
                comp_yvar = comp_file['analysis/'][yvar][:]
            N_merge = np.sum(np.logical_not(np.isnan(merge_yvar))*1)
            N_comp = np.sum(np.logical_not(np.isnan(comp_yvar))*1)
            compmeans = np.full(11, np.nan)
            mergemeans = np.full(11, np.nan)
            for i in np.arange(1,11):
                binind_merge = (binned_merge == i)
                binind_comp = (binned_comp == i)
                if np.sum(binind_merge*1) > 10:
                    compmeans[i-1] = np.nanmedian(comp_yvar[binind_comp])
                    mergemeans[i-1] = np.nanmedian(merge_yvar[binind_merge])
            fig, ax1 = plt.subplots(1,1, figsize=(6,5))
            ax1.scatter(x=merge_xvar,y=merge_yvar, alpha=0.5, s=10, color='royalblue')
            ax1.scatter(x=comp_xvar,y=comp_yvar, alpha=0.5, s=10, color='lightcoral')
            if apply_ppf:
                ind = ~np.isnan(compmeans)
                rho_comp = np.corrcoef(cv_bins[ind], compmeans[ind])[0,1]
                ax1.plot(cv_bins[ind], compmeans[ind], '--sr', alpha=0.8, label=complbl + "(ppf)")
                ind = ~np.isnan(mergemeans)
                rho_merge = np.corrcoef(cv_bins[ind],mergemeans[ind])[0,1]
                ax1.plot(cv_bins[ind], mergemeans[ind], '--ob', alpha=0.8, label=mergelbl + "(ppf)")
            else:
                ind = ~np.isnan(compmeans)
                rho_comp = np.corrcoef(cv_bins[ind], compmeans[ind])[0,1]
                ax1.plot(cv_bins[ind], compmeans[ind], '--sr', alpha=0.8, label=complbl)
                ind = ~np.isnan(mergemeans)
                rho_merge = np.corrcoef(cv_bins[ind],mergemeans[ind])[0,1]
                ax1.plot(cv_bins[ind], mergemeans[ind], '--ob', alpha=0.8, label=mergelbl)
            ax1.set_ylabel(yvar)
            ax1.set_xlabel(xvar)
            ax1.set_xticks(cv_bins)
            ax1.set_xticklabels(np.round(cv_bins,4), rotation=90)
            ax1.set_xlim([np.min(cv_bins[~np.isnan(mergemeans)]), np.max(cv_bins[~np.isnan(mergemeans)])])
            yvar_std = np.nanstd(merge_yvar)
            yvar_med = np.nanmedian(merge_yvar)
            ax1.set_ylim([yvar_med - yvar_std, yvar_med + yvar_std])
            ax1.legend()
            plt.title(r'N$_{%s}$ = %d, N$_{%s}$ = %d, $\rho_{%s}$ = %.2f, $\rho_{%s}$ = %.2f'\
                      %(complbl, N_comp, mergelbl, N_merge, complbl, rho_comp, mergelbl, rho_merge))
            if apply_ppf:
                plt.savefig(plot_directory + '/medians_filtered-' + xvar + '_vs_' + yvar)
            else:
                plt.savefig(plot_directory + '/medians-' + xvar + '_vs_' + yvar)
            plt.close()


def plot_scatter(merge_file, comp_file, xvars, yvars, mergelbl='', complbl='', plot_directory='./', apply_ppf=False):
    '''

    '''
    if apply_ppf:
        ind_ppf_merge = post_processing_filter0(merge_file)
        ind_ppf_comp = post_processing_filter0(comp_file)

    for xvar in xvars:
        if apply_ppf:
            merge_xvar = merge_file['analysis/'][xvar][ind_ppf_merge]
            comp_xvar = comp_file['analysis/'][xvar][ind_ppf_comp]
        else:
            merge_xvar = merge_file['analysis/'][xvar][:]
            comp_xvar = comp_file['analysis/'][xvar][:]
        xvar_std = np.nanstd(merge_xvar)
        xvar_med = np.nanmedian(merge_xvar)
        xvar_plotmax = xvar_med+xvar_std*5
        xvar_plotmin = np.nanmin(merge_xvar)
        cv_bins = np.linspace(xvar_plotmin, xvar_plotmax, 11)
        binned_merge = np.digitize(merge_xvar, bins=cv_bins)
        binned_comp = np.digitize(comp_xvar, bins=cv_bins)
        for yvar in yvars:
            if apply_ppf:
                merge_yvar = merge_file['analysis/'][yvar][ind_ppf_merge]
                comp_yvar = comp_file['analysis/'][yvar][ind_ppf_comp]
            else:
                merge_yvar = merge_file['analysis/'][yvar][:]
                comp_yvar = comp_file['analysis/'][yvar][:]
            N_merge = np.sum(np.logical_not(np.isnan(merge_yvar))*1)
            N_comp = np.sum(np.logical_not(np.isnan(comp_yvar))*1)
            compstd = np.full(11, np.nan)
            mergestd = np.full(11, np.nan)
            for i in np.arange(1,11):
                binind_merge = (binned_merge == i)
                binind_comp = (binned_comp == i)
                if np.sum(binind_merge*1) >= 10:
                    compstd[i-1] = np.nanstd(comp_yvar[binind_comp])
                    mergestd[i-1] = np.nanstd(merge_yvar[binind_merge])
            ind = np.logical_not(np.isnan(compstd))
            rho_comp = np.corrcoef(cv_bins[ind],compstd[ind])[0,1]
            ind = np.logical_not(np.isnan(mergestd))
            rho_merge = np.corrcoef(cv_bins[ind],mergestd[ind])[0,1]
            fig, ax1 = plt.subplots(1,1, figsize=(6,5))
            if apply_ppf:
                ax1.plot(cv_bins, compstd, '--sr', alpha=0.8, label=complbl + "(ppf)")
                ax1.plot(cv_bins, mergestd, '--ob', alpha=0.8, label=mergelbl + "(ppf)")
            else:
                ax1.plot(cv_bins, compstd, '--sr', alpha=0.8, label=complbl)
                ax1.plot(cv_bins, mergestd, '--ob', alpha=0.8, label=mergelbl)
            ax1.set_ylabel(yvar + ' scatter')
            ax1.set_xlabel(xvar)
            ax1.set_xticks(cv_bins)
            ax1.set_xticklabels(np.round(cv_bins,4), rotation=90)
            ax1.set_xlim([np.min(cv_bins[~np.isnan(mergestd)]), np.max(cv_bins[~np.isnan(mergestd)])])
            ax1.legend()
            plt.title(r'N$_{%s}$ = %d, N$_{%s}$ = %d, $\rho_{%s}$ = %.2f, $\rho_{%s}$ = %.2f'\
                      %(complbl, N_comp, mergelbl, N_merge, complbl, rho_comp, mergelbl, rho_merge))
            if apply_ppf:
                plt.savefig(plot_directory + '/scatter_filtered-' + xvar + '_vs_' + yvar)
            else:
                plt.savefig(plot_directory + '/scatter-' + xvar + '_vs_' + yvar)
            plt.close()

def plot_medians_ppf(merge_file, xvars, yvars, mergelbl='', plot_directory='./', filteredlbl='ppf'):
    '''

    '''
    for xvar in xvars:
        selected_xvar = merge_file['analysis/'][xvar][:]
        xvar_std = np.nanstd(selected_xvar)
        xvar_med = np.nanmedian(selected_xvar)
        xvar_plotmax = xvar_med+xvar_std*5
        xvar_plotmin = np.nanmin(selected_xvar)
        cv_bins = np.linspace(xvar_plotmin, xvar_plotmax, 11)
        binned_merge = np.digitize(selected_xvar, bins=cv_bins)
        ind_ppf = post_processing_filter0(merge_file)
        filtered_xvar = selected_xvar[ind_ppf]
        binned_filtered = np.digitize(filtered_xvar, bins=cv_bins)
        for yvar in yvars:
            selected_yvar = merge_file['analysis/'][yvar][:]
            N_merge = np.sum(np.logical_not(np.isnan(selected_yvar))*1)
            filtered_yvar = selected_yvar[ind_ppf]
            N_filtered = np.sum(np.logical_not(np.isnan(filtered_yvar))*1)
            filteredmeans = np.full(11, np.nan)
            mergemeans = np.full(11, np.nan)
            for i in np.arange(1,11):
                binind_merge = (binned_merge == i)
                binind_filtered = (binned_filtered ==i)
                if np.sum(binind_filtered*1) > 10:
                    filteredmeans[i-1] = np.nanmedian(filtered_yvar[binind_filtered])
                    mergemeans[i-1] = np.nanmedian(selected_yvar[binind_merge])
            fig, ax1 = plt.subplots(1,1, figsize=(6,5))
            ind = ~np.isnan(filteredmeans)
            rho_filtered = np.corrcoef(cv_bins[ind], filteredmeans[ind])[0,1]
            ax1.plot(cv_bins[ind], filteredmeans[ind], '--sr', alpha=0.8, label=filteredlbl, zorder=3)
            ind = ~(np.isnan(filtered_xvar) | np.isnan(filtered_yvar))
            ax1.scatter(x=filtered_xvar[ind],y=filtered_yvar[ind],alpha=0.8,s=10, color='lightcoral', zorder=2)
            ind = ~np.isnan(mergemeans)
            rho_merge = np.corrcoef(cv_bins[ind], mergemeans[ind])[0,1]
            ax1.plot(cv_bins[ind], mergemeans[ind], '--ob', alpha=0.8, label=mergelbl, zorder=4)
            ind = ~(np.isnan(selected_xvar) | np.isnan(selected_yvar))
            ax1.scatter(x=selected_xvar[ind],y=selected_yvar[ind],alpha=0.4,s=10, color='royalblue', zorder=1)
            ax1.set_ylabel(yvar)
            ax1.set_xlabel(xvar)
            ax1.set_xticks(cv_bins)
            ax1.set_xticklabels(np.round(cv_bins,4), rotation=90)
            ax1.set_xlim([np.min(cv_bins[~np.isnan(mergemeans)]), np.max(cv_bins[~np.isnan(mergemeans)])])
            yvar_std = np.nanstd(selected_yvar)
            yvar_med = np.nanmedian(selected_yvar)
            ax1.set_ylim([yvar_med - yvar_std, yvar_med + yvar_std])
            ax1.legend()
            plt.title(r'N$_{%s}$ = %d, N$_{%s}$ = %d, $\rho_{%s}$ = %.2f, $\rho_{%s}$ = %.2f'\
                      %(filteredlbl, N_filtered, mergelbl, N_merge, filteredlbl, rho_filtered, mergelbl, rho_merge))
            plt.savefig(plot_directory + '/medians_ppf_' + mergelbl + '_' + xvar + '_vs_' + yvar)
            plt.close()


def plot_scatter_ppf(merge_file, xvars, yvars, mergelbl='', plot_directory='./', filteredlbl='ppf'):
    '''

    '''

    for xvar in xvars:
        selected_xvar = merge_file['analysis/'][xvar][:]
        xvar_std = np.nanstd(selected_xvar)
        xvar_med = np.nanmedian(selected_xvar)
        xvar_plotmax = xvar_med+xvar_std*5
        xvar_plotmin = np.nanmin(selected_xvar)
        cv_bins = np.linspace(xvar_plotmin, xvar_plotmax, 11)
        binned_merge = np.digitize(selected_xvar, bins=cv_bins)
        ind_ppf = post_processing_filter0(merge_file)
        filtered_xvar = selected_xvar[ind_ppf]
        binned_filtered = np.digitize(filtered_xvar, bins=cv_bins)
        for yvar in yvars:
            selected_yvar = merge_file['analysis/'][yvar][:]
            N_merge = np.sum(np.logical_not(np.isnan(selected_yvar))*1)
            filtered_yvar = selected_yvar[ind_ppf]
            N_filtered = np.sum(np.logical_not(np.isnan(filtered_yvar))*1)
            filteredstd = np.full(11, np.nan)
            mergestd = np.full(11, np.nan)
            for i in np.arange(1,11):
                binind_merge = (binned_merge == i)
                binind_filtered = (binned_filtered == i)
                if np.sum(binind_filtered*1) >= 10:
                    filteredstd[i-1] = np.nanstd(filtered_yvar[binind_filtered])
                    mergestd[i-1] = np.nanstd(selected_yvar[binind_merge])

            fig, ax1 = plt.subplots(1,1, figsize=(6,5))
            ind = np.logical_not(np.isnan(filteredstd))
            rho_filtered = np.corrcoef(cv_bins[ind],filteredstd[ind])[0,1]
            ax1.plot(cv_bins[ind], filteredstd[ind], '--sr', alpha=0.8, label=filteredlbl)
            ind = np.logical_not(np.isnan(mergestd))
            rho_merge = np.corrcoef(cv_bins[ind],mergestd[ind])[0,1]            
            ax1.plot(cv_bins[ind], mergestd[ind], '--ob', alpha=0.8, label=mergelbl)
            ax1.set_ylabel(yvar + ' scatter')
            ax1.set_xlabel(xvar)
            ax1.set_xticks(cv_bins)
            ax1.set_xticklabels(np.round(cv_bins,4), rotation=90)
            ax1.set_xlim([np.min(cv_bins[~np.isnan(mergestd)]), np.max(cv_bins[~np.isnan(mergestd)])])
            ax1.legend()
            plt.title(r'N$_{%s}$ = %d, N$_{%s}$ = %d, $\rho_{%s}$ = %.2f, $\rho_{%s}$ = %.2f'\
                      %(filteredlbl, N_filtered, mergelbl, N_merge, filteredlbl, rho_filtered, mergelbl, rho_merge))
            plt.savefig(plot_directory + '/scatter_ppf-' + mergelbl + '_' + xvar + '_vs_' + yvar)
            plt.close()


def plot_spatial(merge_file, comp_file, lon_grid, lat_grid, site, mergelbl='', complbl='', plot_directory='./'):
    for key in merge_file['analysis/'].keys():
        fig =  plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1,2,1)
        merge_var = merge_file['analysis/'][key][:]
        comp_var = comp_file['analysis/'][key][:]
        vlim = np.max([np.nanstd(merge_var), np.nanstd(comp_var)])
        vmed = np.max([np.nanmedian(merge_var), np.nanmedian(comp_var)])
        mesh1 = ax1.pcolormesh(lon_grid, lat_grid, merge_var.reshape(lon_grid.shape),
                               vmin=-2*vlim + vmed,
                               vmax=2*vlim + vmed, cmap=cmap_hotr)
        ax1.add_patch( PolygonPatch(polygon=sgeom.Point(site).buffer(.015), color='purple',alpha=1, zorder=6))
        ax1.set_title(key + ', ' + mergelbl)
        ax1.set_ylabel('lat')
        ax1.set_xlabel('lon')
        ax2 = fig.add_subplot(1,2,2)
        mesh2 = ax2.pcolormesh(lon_grid, lat_grid, comp_var.reshape(lon_grid.shape),
                               vmin=-2*vlim + vmed,
                               vmax=2*vlim + vmed, cmap=cmap_hotr)
        ax2.add_patch( PolygonPatch(polygon=sgeom.Point(site).buffer(.015), color='purple',alpha=1, zorder=6))
        ax2.set_title(key + ', ' + complbl)
        ax2.set_xlabel('lon')
        plt.subplots_adjust(right=0.9)
        cax = plt.axes([ 0.905, 0.2, 0.01, 0.6])
        cbar = plt.colorbar(mesh2,cax=cax)
        cbar.set_label(key, rotation=270, labelpad=10)
        plt.savefig(plot_directory + '/spatial-' + key)
        plt.close()

def plot_spatial_anomaly(merge_file, comp_file, lon_grid, lat_grid, site, mergelbl='', complbl='', plot_directory='./'):
    for key in merge_file['analysis/'].keys():
        fig =  plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1,2,1)
        merge_var = merge_file['analysis/'][key][:]
        comp_var = comp_file['analysis/'][key][:]
        vlim = np.nanstd(merge_var)
        vmed = np.nanmedian(merge_var)
        mesh1 = ax1.pcolormesh(lon_grid, lat_grid, merge_var.reshape(lon_grid.shape) - vmed,
                               vmin=-2*vlim,
                               vmax=2*vlim, cmap=cmap_seis)
        ax1.add_patch( PolygonPatch(polygon=sgeom.Point(site).buffer(.015), color='green',alpha=1, zorder=6))
        ax1.set_title(key + ', ' + mergelbl)
        ax1.set_ylabel('lat')
        ax1.set_xlabel('lon')
        cbar = plt.colorbar(mesh1, ax=ax1)
        cbar.set_label('dist from ' + str(round(vmed,2)), rotation=270, labelpad=10)
        ax2 = fig.add_subplot(1,2,2)
        vlim = np.nanstd(comp_var)
        vmed = np.nanmedian(comp_var)
        mesh2 = ax2.pcolormesh(lon_grid, lat_grid, comp_var.reshape(lon_grid.shape) - vmed,
                               vmin=-2*vlim,
                               vmax=2*vlim, cmap=cmap_seis)
        ax2.add_patch( PolygonPatch(polygon=sgeom.Point(site).buffer(.015), color='green',alpha=1, zorder=6))
        ax2.set_title(key + ', ' + complbl)
        ax2.set_xlabel('lon')
        cbar = plt.colorbar(mesh2, ax=ax2)
        cbar.set_label('dist from ' + str(round(vmed,2)), rotation=270, labelpad=10)
        plt.savefig(plot_directory + '/spatial-anomaly-' + key)
        plt.close()

def plot_kde(merge_file, comp_file, mergelbl='', complbl='', plot_directory='./', centered=True):
    for key in merge_file['analysis/'].keys():
        merge_key = merge_file['analysis/'][key][np.logical_not(np.isnan(merge_file['analysis/'][key][:]))]
        N_merge = np.size(merge_key)
        mu_merge = np.nanmedian(merge_key)
        sd_merge = np.nanstd(merge_key)
        z_merge = (merge_key-mu_merge) if centered else merge_key
        fig =  plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(1,1,1)
        sns.kdeplot(z_merge, shade=True, ax=ax1, label=r'w/ sh, N = %d, $\mu$ = %.4e, $\sigma$ = %.4e'%( N_merge, mu_merge, sd_merge))
        comp_key = comp_file['analysis/'][key][np.logical_not(np.isnan(comp_file['analysis/'][key][:]))]
        N_comp = np.size(comp_key)
        mu_comp = np.nanmedian(comp_key)
        sd_comp = np.nanstd(comp_key)
        z_comp = (comp_key-mu_comp) if centered else comp_key
        sns.kdeplot(z_comp, shade=True, ax=ax1, label=r'w/o sh, N = %d, $\mu$ = %.4e, $\sigma$ = %.4e'%( N_comp, mu_comp, sd_comp))
        max_sd = np.max([sd_merge, sd_comp])
        plt.xlim([mu_merge-4*max_sd, mu_merge+4*max_sd])
        plt.legend()
        fig.tight_layout()
        ax1.set_title( (key + ' (centered)' if centered else key ) )
        plt.savefig(plot_directory + '/hist-anomaly_' + key)
        plt.close()

def plot_error_correlation(merge_file, keys, plot_directory='./'):
    ind = np.logical_not(np.isnan(merge_file['analysis/']['xco2'][:]))
    _dat = np.zeros([sum(ind*1), len(keys)])
    for i in range(len(keys)):
        key = keys[i]
        _dat[:,i] = merge_file['analysis/'][key][ind]
    # pairs data with labels for easy sorting
    df = pd.DataFrame(_dat, columns=keys)
    corrmat = df.corr()
    spearmat = df.corr('spearman')
    N = len(keys)
    for i in np.arange(1,5):
        temp_pearson = corrmat[-i:N-i+1].sort_values(keys[-i],1).T
        temp_pearson = temp_pearson.drop(gas_keys,0)
        temp_spearman = spearmat[-i:N-i+1].sort_values(keys[-i],1).T
        temp_spearman = temp_spearman.drop(gas_keys,0)
        fig = plt.figure(figsize=(4,9), dpi=75)
        ax1 = fig.add_subplot(1,2,1)
        sns.heatmap(temp_pearson, cmap='seismic', ax=ax1, yticklabels=True, cbar=False,annot=True, vmin=-1, vmax=1)
        ax1.set_title('Pearson')
        ax2 = fig.add_subplot(1,2,2)
        sns.heatmap(temp_spearman, cmap='seismic', ax=ax2, yticklabels=True, cbar=False,annot=True, vmin=-1, vmax=1)
        ax2.yaxis.tick_right()
        plt.yticks(rotation=0)
        ax2.set_title('Spearman')
        plt.savefig(plot_directory + '/correlation_gas-' + keys[-i], bbox_inches=None)
        plt.close()

def post_processing_filter0(merge_file):
    '''
    PPF as described in Polonsky et al. (2014).
    '''
    chi0 = merge_file['analysis/chi2_o2'][:]
    chi1 = merge_file['analysis/chi2_weak_co2'][:]
    chi2 = merge_file['analysis/chi2_strong_co2'][:]
    chi3 = merge_file['analysis/chi2_ch4'][:]
    chisum = chi0 + chi1 + chi2 + chi3
    dofco2 = merge_file['analysis/dof_co2_profile'][:]
    ind = ( (chisum < 2) & (dofco2 > 1.6))
    return ind 

