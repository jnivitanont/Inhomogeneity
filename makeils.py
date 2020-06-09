#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Authors:     Eric Burgh (LM-ATC), Sean Crowell (OU), Jeff Nivitanont (OU)
# @Description: This script models the way light diffracts in the GeoCarb instrument, and calculates the resulting Instrument Line Shape (ILS).
# @Output:      None. The subsampled albedo file is modified inplace to contain the resulting ILS's.

import numpy as np
from scipy.special import jv as besselj
from scipy.ndimage.filters import convolve as convol
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import pdb
import math
from numba import jit
from h5py import File
from netCDF4 import Dataset
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import matplotlib.cm as cmx
import argparse

parser = argparse.ArgumentParser(description='Calculate ILS using subsampled MODIS albedos')
parser.add_argument('-al','--albedo', metavar='albedo file', required=True, help='Albedo File')
parser.add_argument('-ar','--arp', metavar='arp file', required=True, help='ARP file')
args = parser.parse_args()

alb_fid=args.albedo #subsampled albedo file. Output of subsample_gc_footprints.py
arp_fid=args.arp #radiometric file containing spectral gridding information
trf_fid="ReplannedSHs.h5" #This file contains the transfer function that characterizes a slit homogenizer's performance
dtor = np.pi/180.
plot_ils = False

@jit
def robust_mean(Y,cut):
    '''
    NAME:
       Robust_Mean 

    PURPOSE:
       Outlier-resistant determination of the mean and standard deviation.

    EXPLANATION:
       Robust_Mean trims away outliers using the median and the median
       absolute deviation.    An approximation formula is used to correct for
       the trunction caused by trimming away outliers

    CALLING SEQUENCE:
       mean = Robust_Mean( VECTOR, Sigma_CUT, Sigma_Mean, Num_RejECTED)

    INPUT ARGUMENT:
          VECTOR    = Vector to average
          Sigma_CUT = Data more than this number of standard deviations from the
                  median is ignored. Suggested values: 2.0 and up.

    OUTPUT ARGUMENT:
          Mean  = the mean of the input vector, numeric scalar

    KEYWORDS:

          GoodInd = The indices of the values not rejected

    OPTIONAL OUTPUTS:
    Sigma_Mean = the approximate standard deviation of the mean, numeric
               scalar.  This is the Sigma of the distribution divided by sqrt(N-1)
               where N is the number of unrejected points. The larger
               SIGMA_CUT, the more accurate. It will tend to underestimate the
               true uncertainty of the mean, and this may become significant for
               cuts of 2.0 or less.
          Num_RejECTED = the number of points trimmed, integer scalar

    EXAMPLE:
          IDL> a = randomn(seed, 10000)    #Normal distribution with 10000 pts
          IDL> Robust_Mean,a, 3, mean, meansig, num    #3 Sigma clipping   
          IDL> print, mean, meansig,num

          The mean should be near 0, and meansig should be near 0.01 ( =
           1/sqrt(10000) ).    
    PROCEDURES USED:
          AVG() - compute simple mean
    REVISION HISTORY:
          Written, H. Freudenreich, STX, 1989# Second iteration added 5/91.
          Use MEDIAN(/EVEN)    W. Landsman   April 2002
          Correct conditional test, higher order truncation correction formula
                   R. Arendt/W. Landsman   June 2002
          New truncation formula for sigma H. Freudenriech  July 2002
    '''
  
    Npts    = len(Y)
    YMed    = np.median(Y)
    AbsDev  = np.abs(Y-YMed)
    MedAbsDev = np.median(AbsDev)/0.6745
    if MedAbsDev < 1.0E-24: MedAbsDev = AbsDev.mean()/.8  
    Cutoff  = cut*MedAbsDev  
    GoodInd = np.where(AbsDev < Cutoff)[0]
    GoodPts = Y[ GoodInd ]
    Mean    = GoodPts.mean()
    Sigma   = GoodPts.std()
    # Compenate Sigma for truncation (formula by HF):
    if cut < 4.50: Sigma=Sigma/(-0.15405+0.90723*cut-0.23584*cut**2+0.020142*cut**3)  
    # Now the standard deviation of the mean:
    Sigma = Sigma/math.sqrt(Npts-1.)  
    return Mean, Sigma

@jit
def gratinglambda(sigma, alpha, beta, gamma=0, m=1):
    #alpha, beta, gamma in degrees
    #sigma is mm/line
    lmb = sigma/m*(np.sin(alpha*dtor)+np.sin(beta*dtor))*np.cos(gamma*dtor)
    #lambda will be returned in same units as sigma
    return lmb

@jit
def get_geocarb_gratinginfo(band):

    '''
    ++++++++++++++++++++++++
    INPUTS:

       band:   GeoCarb band identification, following:
               0 = O2A Band (0.765 microns)
               1 = WCO2 Band (1.606 microns)
               2 = SCO2 Band (2.06 microns)
               3 = CH4/CO Band (2.32 microns)

    OUTPUTS:

       sigma:  line spacing (inverse of line ruling) in microns
       alpha:  angle of incidence in degrees
       beta0:  angle of diffraction for nominal central wavelength in degrees
       order:  order used for the specific band
       fcam:   focal length of camera in mm
    -------------------------
    '''
    sigmas = 1e3/np.array([107.08,107.08,91.7,91.7]) #microns #NUMBERS FROM CATHY FOR LINE RULING
    alphas = np.array([53.9, 53.9, 53.0, 53.0]) #degrees
    beta0s = np.array([65.8, 65.8, 64.9, 64.9])
    orders = np.array([21, 10, 9, 8])
    fcams = np.array([220.102,222.519,220.816,218.457])

    sigma = sigmas[band]
    alpha = alphas[band]
    beta0 = beta0s[band]
    order = orders[band]
    fcam = fcams[band]

    return sigma,alpha,beta0,order,fcam

@jit
def betaangle(cenwave,sigma,alpha,gamma=0,m=1):
    beta=np.arcsin(m*cenwave/sigma/np.cos(gamma*dtor)-np.sin(alpha*dtor))/dtor
    return beta

def pointils2(band,wave):
    #VERSION 2 updates for replanned optics (smaller grating footprint)
    gratingsizes = np.array([81., 81., 84.4, 84.4])
    #make function to generate pointils.
    #convolution of grating function with airy function for the relevant band
    deltawave = 1e-6
    [sigma,alpha,beta0,order,fcam] = get_geocarb_gratinginfo(band)
    gratingsize=gratingsizes[band]
    #find central wavelength
    cenwave = 0.5*(wave[len(wave)//2]+wave[len(wave)//2+1])#gratinglambda(sigma,alpha,beta0,m=order)
    #wave=np.arange(0.001*2/deltawave)*deltawave+cenwave-0.001
    #compute beta angles for these wavelengths
    betas = betaangle(wave,sigma,alpha,m=order)

    #FIRST DO GRATING FUNCTION
    #number of illuminated grooves
    Ngrooves = gratingsize*1000./sigma
    #phase shift
    deltaphi = 2*np.pi*sigma/cenwave*(np.sin(betas*dtor)-np.sin(beta0*dtor))
    #total phase shift across grating
    phi = Ngrooves*deltaphi
    inten = 1/Ngrooves**2*(np.sin(phi/2)/np.sin(deltaphi/2))**2
    deltawave = wave-cenwave

    #NOW FOR AIRY FUNCTION
    k  = 2*np.pi/cenwave
    ap = 75./2./2.                   #radius of aperture in mm (extra factor of two from descope)
    bx = k*ap*1000.*np.sin((betas-beta0)*dtor)
    #take into account that beam speed in spectral direction
    #has changed due to grating magnification
    bx = bx*np.cos(beta0*dtor)/np.cos(alpha*dtor)
    airy = (2*besselj(1,bx)/bx)**2
    #pdb.set_trace()
    airy = airy/np.nanmax(airy)
    #diffraction limit FWHM
    diffFWHM = cenwave*3.2*np.sqrt(2)*np.cos(alpha*dtor)/np.cos(beta0*dtor)   

    #POINT ILS IS CONVOLUTION OF GRATING FUNCTION WITH AIRY FUNCTION
    pointils = convolve1d(inten,airy, mode='constant', cval=0.0)
    #pdb.set_trace()
    pointils = pointils/pointils.max()
    return pointils


def makeils4(band,inputslit,resolving_power=0,transfer=0,ils_grid=[]):
    '''
    #VERSION 2 makes this a function and allows for pointils to be
    #fabricated outside of this function
  
    #make function to generate full ils including slit, pointils, and
    #spectrograph image quality.
    #
    #this still assumes center of band only

    #VERSION 3 makes slit be defined outside and allows Slit Homogenizer
    #transfer function to be applied

    #VERSION 4 uses SH transfer functions made using replanned optical
    #design and pointils2.pro


    #INPUTS:
    #    band:   GeoCarb band identification, following:
    #            0 = O2A Band (0.765 microns)
    #            1 = WCO2 Band (1.606 microns)
    #            2 = SCO2 Band (2.06 microns)
    #            3 = CH4/CO Band (2.32 microns)
    #
    #    wave:   Optional wavelength grid upon which to calculate the
    #            ILS. If not specified, default will be a wavelength array
    #            that is 20000 elements long with wavelength centered on
    #            band and wavelength per pixel of 1e-7 microns 
    '''

    deltawave = 1e-6 
    #assume error budget is 35 microns
    slitwidth = 36.                 #microns
    #slitwidth = 27.45               #microns 
    [sigma,alpha,beta0,order,fcam] = get_geocarb_gratinginfo(band)
    #find central wavelength
    cenwave = gratinglambda(sigma,alpha,beta0,m=order)
    wave=np.arange(-int(0.001*2/deltawave)/2-0.5,int(0.001*2/deltawave)/2+1)*deltawave + cenwave

    #compute beta angles for these wavelengths
    betas = betaangle(wave,sigma,alpha,m=order)
    pointils = pointils2(band,wave)

    #linear position at the detector
    dx = (betas-beta0)*dtor*fcam*1000.

    #ALLOW FOR SLIT FUNCTION TO BE DEFINED BEFOREHAND. THIS ALLOWS FOR
    #INHOMOGENEOUS SLIT ILLUMINATIONS

    #NOW COMPUTE SLIT FUNCTION
    gratingmag = np.cos(alpha*dtor)/np.cos(beta0*dtor)
    dxslit = slitwidth*gratingmag    #magnified by grating
    inslit = np.where(abs(dx) <= dxslit/2)[0]
    detslit = np.zeros(len(dx))
    detslit[inslit]=1.

    #INPUT SLIT FUNCTION HAS BEEN DEFINED ON 0 TO 1 SCALE for scale of -27
    #microns to +27 microns at cross-slit telescope focal plane 
    #so need new dx that scales as 0 to 1 across imaged slit width
    inputx = slitwidth*(np.linspace(0,1,len(inputslit))-0.5)
    inputxsh = 1.5*inputx

    if transfer:
        transferf= File(trf_fid, 'r')['arrays'][band,:,:]
        zin = File(trf_fid, 'r')['zin'][:]
        zout = File(trf_fid, 'r')['zout'][:]
        shinput = np.interp(zin,inputxsh,inputslit)
        shoutput = np.matmul(transferf,shinput.T)
        f_interp = interp1d(zout*gratingmag,shoutput,fill_value=0.,bounds_error=False)
        slit = f_interp(dx)
        nz = np.where(slit > 0)
    else:
        #NOW MAKE SLIT BEFORE USE OF SLIT HOMOGENIZER in case /transfer not used
        baseslit = np.zeros(len(inputxsh))
        for ix,x in enumerate(inputxsh):
            if (x >= inputx.min())*(x <= inputx.max()):
                baseslit[ix] = np.interp(inputxsh[ix],inputx,inputslit)
        slit = np.interp(dx,inputxsh*gratingmag,baseslit)
    #NOW COMPUTE SPECTROGRAPH IMAGE QUALITY
    #FOR NOW ASSUME DIFFRACTION LIMIT
    if resolving_power > 0:
        specim = np.exp(-dx**2/(cenwave/resolving_power/np.sqrt(np.log(2))**2))
        specim = specim/np.max(specim)
    #NOW GENERATE PIXEL FUNCTION
    #pixels are 18 microns
    pix = np.zeros(len(dx))
    inpix = np.where(abs(dx) <= 9.)[0]
    pix[inpix]=1.
    #NOW START THE CONVOLUTIONS
    #FIRST CONVOLVE SLIT WITH POINTILS
    ils0 = convolve1d(slit,pointils,mode='constant',cval=0.0)#conv_circ(slit,pointils)#, mode='constant')
    #NEXT CONVOLVE THIS WITH SPECTROGRAPH IMAGE QUALITY
    if resolving_power > 0: 
        ils1 = convolve1d(ils0,specim,mode='constant',cval=0.0)#, mode='constant')
    else: 
        ils1 = ils0[:]
    #NEXT CONVOLVE THIS WITH PIXEL FUNCTION
    ils2 = convolve1d(ils1,pix,mode='constant',cval=0.0)#conv_circ(ils1,pix)#, mode='constant')
    out_wave = wave[:]
    if len(ils_grid) > 0:
        out_wave = cenwave+ils_grid
    ils_g = np.interp(out_wave,wave,ils2)
    return out_wave-cenwave,ils_g,dx,slit

def create_theoretical_ils():
    # read in the ILS spectral grid
    fid = File(arp_fid,'r')
    dlam = fid['SpectralConversion/ils_delta_lambda'][:][:,0,0,:]
    slit_length = 1000 #subslit_alb.shape[-1]

    inslit = {}
    inslit['uniform'] = np.ones(slit_length)
    inslit['point'] = np.zeros(slit_length)+0.01
    inslit['point'][int(slit_length/5)] = 1.
    inslit['Quarter'] = np.ones(slit_length)
    inslit['Quarter'][:int(0.25*slit_length)] = 0.
    inslit['Half'] = np.ones(slit_length)
    inslit['Half'][:int(slit_length/2)] = 0.
    inslit['ThreeQuarter'] = np.ones(slit_length)
    inslit['ThreeQuarter'][:int(0.75*slit_length)] = 0
    inslit['linear'] = np.linspace(0,1,slit_length)
    inslit['subslit'] = np.zeros(slit_length)
    inslit['subslit'][int(0.25*slit_length):int(0.5*slit_length)] = 1
    slit_keys = ['uniform']#,'Quarter','Half','ThreeQuarter']#,'subslit']
    ils = {}
    slit = {}
    for ib,b in zip(range(4),['nir','wco2','sco2','ch4']):
        #plt.figure()
        # if ib < 3:
        dl = dlam[ib]
        # else:
            # dl = dlam[ib-1]
        names = ['uniform','uniform SH']
        for ky in slit_keys:
            names.extend([ky,ky+' SH'])
        ils[b] = {}
        slit[b] = {}
        for ky in slit_keys:
            for ish,sh in enumerate(['no_homog','with_homog']):
                k = ky+'_'+sh
                ils[b][k] = {}
                slit[b][k] = {}
                wave,tils,slit_grid,slit_val = makeils4(ib,inslit[ky]*100.,transfer=ish,ils_grid=dl)
                ils[b][k]['value'] = tils/np.trapz(tils,wave)
                cdf = np.array([np.trapz(ils[b][k]['value'][:i],wave[:i]) for i in range(len(wave))])
                ils[b][k]['offset'] = np.where(cdf <= 0.5)[0][-1]
                slit[b][k]['value'] = slit_val[:]
                slit[b][k]['grid'] = slit_grid[:]
        ils[b]['grid'] = wave[:]
        if plot_ils:

            #jet = cm = plt.get_cmap('jet') 
            #cNorm  = colors.Normalize(vmin=0, vmax=fp_nums[-1])
            #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            fig = plt.figure(figsize=(8,14))
            gs = GridSpec(3,1)
            ax = fig.add_subplot(gs[0,0])

            inds = np.where(slit[b]['uniform_with_homog']['value'] > 0)[0]
            labels = []
            for ky in slit_keys:
                ax.plot(slit[b][ky+'_with_homog']['grid'][inds],slit[b][ky+'_with_homog']['value'][inds]/slit[b]['uniform_with_homog']['value'].max()*100)#,'--',color=scalar)
                labels.append(ky+' SH')
            plt.legend(labels,ncol=len(slit_keys)+1)
            plt.title('Band '+str(b)+' Slit Functions')


            ax = fig.add_subplot(gs[1,0])
            labels = []
            inds = np.where(ils[b]['uniform_with_homog']['value'] >= ils[b]['uniform_with_homog']['value'].max()*0.02)[0]
            for ky in slit_keys:
                ax.plot(wave[inds],ils[b][ky+'_with_homog']['value'][inds])#,ls='--',color=line_colors[ky])#scalarMap.to_rgba(ky),ls='--')
                labels.append(str(ky)+' SH')
            plt.legend(labels,ncol=2)
            plt.title('Band '+str(b)+' Normalized ISRF')

            ax = fig.add_subplot(gs[2,0])
            labels=[]
            for ky in slit_keys:
                ax.plot(wave[inds],(ils[b][ky+'_with_homog']['value'][inds]-ils[b]['uniform_with_homog']['value'][inds])/ils[b]['uniform_with_homog']['value'][inds].max()*100.)#,color=line_colors[ky])#scalarMap.to_rgba(ky))
                ax.set_ylabel('% Error')
                ax.set_ylim([-20,20])
                labels.append(str(ky)+' SH')

            #plt.yscale('log')
            plt.legend(labels,ncol=2)
            plt.title('Band '+str(b)+' ISRF Percentage Errors')
            plt.tight_layout()

            plt.savefig('slit_plot_band%s.png'%b,bbox_inches='tight')
            plt.show()

    # return ils

def create_modis_derived_ils():

    # read in the ILS spectral grid
    fid = Dataset(alb_fid,'a')
    dlam = fid['ils_delta_lambda'][:]
    y = fid['gc_subslit_alb_band2'][:]
    subslit_alb = np.zeros((4,y.shape[0],y.shape[1]))
    subslit_alb[0] = y[:]
    subslit_alb[1] = fid['gc_subslit_alb_band6'][:]
    subslit_alb[2] = fid['gc_subslit_alb_band7'][:]
    subslit_alb[3] = fid['gc_subslit_alb_band7'][:]

    slit_length = subslit_alb.shape[-1]
    n_slits = subslit_alb.shape[0]
    fp_nums = range(subslit_alb.shape[1])

    inslit = {}
    inslit['uniform'] = np.ones(slit_length)
    slit_keys = [str(i) for i in fp_nums]
    for b in range(3):
        names = ['uniform','uniform SH']
        for fp,ky in zip(fp_nums,slit_keys):
            names.extend([ky,ky+' SH'])
            inslit[fp] = subslit_alb[b,fp,:]/subslit_alb[b,fp,:].mean()
        ils = {}
        slit = {}
        ky = 'uniform'
        ils[ky] = {}
        slit[ky] = {}
        for ish,sh in enumerate(['no_homog','with_homog']):
            wave,tils,slit_grid,slit_val = makeils4(b,inslit[ky],transfer=ish,ils_grid=dlam[b])
            ils[ky][sh] = {}
            ils[ky][sh]['value'] = tils/np.trapz(tils,wave)
            ils[ky][sh]['grid'] = wave[:]
            slit[ky][sh] = {}
            slit[ky][sh]['value'] = slit_val[:]
            slit[ky][sh]['grid'] = slit_grid[:]
        for ky in fp_nums:
            ils[ky] = {}
            slit[ky] = {}
            for ish,sh in enumerate(['no_homog','with_homog']):
                wave,tils,slit_grid,slit_val = makeils4(b,inslit[ky],transfer=ish,ils_grid=dlam[b])
                ils[ky][sh] = {}
                ils[ky][sh]['value'] = tils/np.trapz(tils,wave)
                ils[ky][sh]['grid'] = wave[:]
                ils[ky][sh]
                slit[ky][sh] = {}
                slit[ky][sh]['value'] = slit_val[:]
                slit[ky][sh]['grid'] = slit_grid[:]

        if plot_ils:
            plt.figure()
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=0, vmax=fp_nums[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            fig = plt.figure(figsize=(8,10))
            gs = GridSpec(3,1)
            ax = fig.add_subplot(gs[0,0])

            ky = 'uniform'
            labels = []
            lstl = ['','--']
            homog_names = ['',' SH']
            for ish,sh in enumerate(['no_homog','with_homog']):
                inds = np.where(slit[ky][sh]['value'] > 0)[0]
                ax.plot(slit[ky][sh]['grid'][inds],slit[ky][sh]['value'][inds]/slit[ky][sh]['value'][inds].mean(),color=line_colors[ky],ls=lstl[ish])
                labels.append('Uniform'+homog_names[ish])
            for ky in fp_nums:
                for ish,sh in enumerate(['no_homog','with_homog']):
                    ax.plot(inslit[ky],color=scalarMap.to_rgba(ky))
                    ax.plot(slit[ky][sh]['grid'][inds],slit[ky][sh]['value'][inds],ls=lstl[ish],color=scalarMap.to_rgba(ky))
                    labels.append(str(ky)+homog_names[ish])
            plt.legend(labels,ncol=len(slit_keys)+1)
            plt.title('Band '+str(b)+' Slit Functions')

            ax = fig.add_subplot(gs[1,0])
            labels = []
            inds = np.where(ils['uniform']['with_homog']['value'] >= ils['uniform']['with_homog']['value'].max()*0.02)[0]
            ky = 'uniform'
            ax.plot(wave[inds],ils[ky]['no_homog'][inds]/(ils[ky]['no_homog'][inds]*np.diff(wave[inds])[0]).sum(),color=line_colors[ky])
            labels.append(ky)
            ax.plot(wave[inds],ils[ky]['with_homog'][inds],color=line_colors[ky],ls='--')
            labels.append(str(ky)+' SH')
            for ky in fp_nums:
                inds = np.where(ils[ky]['no_homog'] >= ils[ky]['no_homog'].max()*0.002)[0]
                ax.plot(wave[inds],ils[ky]['no_homog'][inds]/(ils[ky]['no_homog'][inds]*np.diff(wave[inds])[0]).sum(),color=scalarMap.to_rgba(ky))
                labels.append(ky)
                ax.plot(wave[inds],ils[ky]['with_homog'][inds],ls='--',color=scalarMap.to_rgba(ky))
                labels.append(str(ky)+' SH')
            plt.legend(labels,ncol=2)
            plt.title('Band '+str(b)+' Normalized ISRF for Different Footprints Near Lamont')

            ax = fig.add_subplot(gs[2,0])
            labels=[]
            ky = 'uniform'
            ax.plot(wave[inds],ils[ky]['no_homog'][inds]/(ils[ky]['no_homog'][inds]*np.diff(wave[inds])[0]).sum()-ils['uniform']['no_homog'][inds]/(ils['uniform']['no_homog'][inds]*np.diff(wave[inds])[0]).sum(),color=line_colors[ky])
            labels.append(ky)
            ax.plot(wave[inds],ils[ky]['with_homog'][inds]-ils['uniform']['with_homog'][inds],color=line_colors[ky])
            labels.append(str(ky)+' SH')
            for ky in fp_nums:
                ax.plot(wave[inds],ils[ky]['no_homog'][inds]/(ils[ky]['no_homog'][inds]*np.diff(wave[inds])[0]).sum()-ils['uniform']['no_homog'][inds]/(ils['uniform']['no_homog'][inds]*np.diff(wave[inds])[0]).sum(),color=scalarMap.to_rgba(ky))
                labels.append(ky)
                ax.plot(wave[inds],ils[ky]['with_homog'][inds]-ils['uniform']['with_homog'][inds],color=scalarMap.to_rgba(ky))
                labels.append(str(ky)+' SH')

            #plt.yscale('log')
            plt.legend(labels,ncol=2)
            plt.title('Band '+str(b)+' ISRF Errors')
            plt.tight_layout()

            plt.savefig('modis_slit_plot_band%s.png'%b,bbox_inches='tight')
            plt.show()
    return ils

def create_append_unif_ils():
    fid = File(alb_fid,'a')
    dlam = fid['ils_delta_lambda'][:]
    n_slits, slit_length = fid['gc_subslit_alb_band2'][:].shape
    inslit = np.ones(slit_length)
    for ish,sh in enumerate(['ils_without_sh','ils_with_sh']):
        sh_tag = sh + '_unif'
        try:
            del fid[sh_tag]
            print('Modifying', sh_tag)
        except:
            print('Creating', sh_tag)
        fid[sh_tag] = np.zeros(dlam.shape)
        for b in range(4):
            print('Calulating band', str(b+1), sh_tag)
            wave,tils,slit_grid,slit_val = makeils4(b,inslit,transfer=ish,ils_grid=dlam[b])
            fid[sh_tag][b] = tils/np.trapz(tils,wave)
    fid.close()

def create_append_high_res_ils():
    fid = File(alb_fid,'a')
    dlam = fid['ils_delta_lambda'][:]
    y = fid['gc_subslit_alb_band2'][:]
    subslit_alb = np.zeros((4,y.shape[0],y.shape[1]))
    subslit_alb[0] = y[:]
    subslit_alb[1] = fid['gc_subslit_alb_band6'][:]
    subslit_alb[2] = fid['gc_subslit_alb_band7'][:]
    subslit_alb[3] = fid['gc_subslit_alb_band7'][:]
    n_slits, n_footprints, slit_length = subslit_alb.shape
    for ish,sh in enumerate(['ils_without_sh','ils_with_sh']):
        try:
            del fid[sh]
            print('Modifying', sh)
        except:
            print('Creating', sh)
        fid[sh] = np.zeros((4, n_footprints, dlam.shape[1]))
        for b in range(4):
            print('Calulating band', str(b+1), sh)
            for fp in range(n_footprints):    
                wave,tils,slit_grid,slit_val = makeils4(b,subslit_alb[b,fp,:],transfer=ish,ils_grid=dlam[b])
                fid[sh][b,fp] = tils/np.trapz(tils,wave)
            #endfor
        #endfor
    #endfor
    fid.close()

if __name__ == '__main__':
    create_append_unif_ils()
    create_append_high_res_ils()
