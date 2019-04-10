import numpy as np
from scipy.special import jv as besselj
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
from numba import jit

dtor = np.pi/180.
band = 0
slit_length = 51
names = ['uniform','point','half','linear']

#need to generate range of betas around the order of interest
         

@jit
def robust_mean(Y,cut,Sigma):
#+
# NAME:
#    Robust_Mean 
#
# PURPOSE:
#    Outlier-resistant determination of the mean and standard deviation.
#
# EXPLANATION:
#    Robust_Mean trims away outliers using the median and the median
#    absolute deviation.    An approximation formula is used to correct for
#    the trunction caused by trimming away outliers
#
# CALLING SEQUENCE:
#    mean = Robust_Mean( VECTOR, Sigma_CUT, Sigma_Mean, Num_RejECTED)
#
# INPUT ARGUMENT:
#       VECTOR    = Vector to average
#       Sigma_CUT = Data more than this number of standard deviations from the
#               median is ignored. Suggested values: 2.0 and up.
#
# OUTPUT ARGUMENT:
#       Mean  = the mean of the input vector, numeric scalar
#
# KEYWORDS:
#
#       GoodInd = The indices of the values not rejected
#
# OPTIONAL OUTPUTS:
#Sigma_Mean = the approximate standard deviation of the mean, numeric
#            scalar.  This is the Sigma of the distribution divided by sqrt(N-1)
#            where N is the number of unrejected points. The larger
#            SIGMA_CUT, the more accurate. It will tend to underestimate the
#            true uncertainty of the mean, and this may become significant for
#            cuts of 2.0 or less.
#       Num_RejECTED = the number of points trimmed, integer scalar
#
# EXAMPLE:
#       IDL> a = randomn(seed, 10000)    #Normal distribution with 10000 pts
#       IDL> Robust_Mean,a, 3, mean, meansig, num    #3 Sigma clipping   
#       IDL> print, mean, meansig,num
#
#       The mean should be near 0, and meansig should be near 0.01 ( =
#        1/sqrt(10000) ).    
# PROCEDURES USED:
#       AVG() - compute simple mean
# REVISION HISTORY:
#       Written, H. Freudenreich, STX, 1989# Second iteration added 5/91.
#       Use MEDIAN(/EVEN)    W. Landsman   April 2002
#       Correct conditional test, higher order truncation correction formula
#                R. Arendt/W. Landsman   June 2002
#       New truncation formula for sigma H. Freudenriech  July 2002
#-  
  
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
    #Num_Rej = Npts - len(GoodPts) 
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

@jit
def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    f2 = np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))
    f2 = np.roll(f2, -((f2.shape[0] - 1)//2))
    return f2

@jit
def pointils2(band):
    #VERSION 2 updates for replanned optics (smaller grating footprint)
    gratingsizes = np.array([81., 81., 84.4, 84.4])
    #make function to generate pointils.
    #convolution of grating function with airy function for the relevant band
    deltawave = 1e-7  
    [sigma,alpha,beta0,order,fcam] = get_geocarb_gratinginfo(band)
    gratingsize=gratingsizes[band]
    #find central wavelength
    cenwave = gratinglambda(sigma,alpha,beta0,m=order)
    wave=np.arange(0.001*2/deltawave)*deltawave+cenwave-0.001
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
    airy = airy/airy.max()
    #diffraction limit FWHM
    diffFWHM = cenwave*3.2*np.sqrt(2)*np.cos(alpha*dtor)/np.cos(beta0*dtor)   

    #POINT ILS IS CONVOLUTION OF GRATING FUNCTION WITH AIRY FUNCTION
    pointils = conv_circ(inten,airy)#, mode='wrap')
    pointils = pointils/pointils.max()

    return pointils

@jit
def makeils4(band,inputslit,specimfwhm=0): #transfer=False,
	'''
    VERSION 2 makes this a function and allows for pointils to be
    fabricated outside of this function
  
    make function to generate full ils including slit, pointils, and
    spectrograph image quality.
    
    this still assumes center of band only

    VERSION 3 makes slit be defined outside and allows Slit Homogenizer
    transfer function to be applied

    VERSION 4 uses SH transfer functions made using replanned optical
    design and pointils2.pro


    INPUTS:
       band:   GeoCarb band identification, following:
               0 = O2A Band (0.765 microns)
               1 = WCO2 Band (1.606 microns)
               2 = SCO2 Band (2.06 microns)
               3 = CH4/CO Band (2.32 microns)
    
       wave:   Optional wavelength grid upon which to calculate the
               ILS. If not specified, default will be a wavelength array
               that is 20000 elements long with wavelength centered on
               band and wavelength per pixel of 1e-7 microns 
	'''
    deltawave = 1e-7 
    #assume error budget is 35 microns
    slitwidth = 36.                 #microns    
    [sigma,alpha,beta0,order,fcam] = get_geocarb_gratinginfo(band)
    #find central wavelength
    cenwave = gratinglambda(sigma,alpha,beta0,m=order)
    #pdb.set_trace()
    wave=np.arange(0.001*2/deltawave)*deltawave+cenwave-0.001
    #compute beta angles for these wavelengths
    betas = betaangle(wave,sigma,alpha,m=order)
    pointils = pointils2(band)
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

    inputx = np.linspace(0,1,len(inputslit))
    inputxsh = slitwidth*1.5*(inputx-0.5)

    #pdb.set_trace()
    #NOW MAKE SLIT BEFORE USE OF SLIT HOMOGENIZER in case /transfer not
    #used
    outslit = np.where(abs(inputxsh) > slitwidth/2)[0]
    baseslit=inputslit[:]
    baseslit[outslit]=0.

    slit = np.interp(dx,inputxsh*gratingmag,baseslit)

    # if transfer:
    #     'ReplannedSHs.sav'
    #     transfer= np.loadtxt('ReplannedSHs.txt')
    #     shinput = interp(zin,inputxsh,inputslit)
    #     shoutput = shinput[:] #transfer
    #     slit = np.interp(dx,zout*gratingmag,shoutput)
    #     nz = np.where(slit > 0)
    #     slit = slit/slit[nz].mean()#robust_mean(slit[nz],1)

    #NOW COMPUTE SPECTROGRAPH IMAGE QUALITY
    #FOR NOW ASSUME DIFFRACTION LIMIT

    if specimfwhm > 0:
        specim = signal.gaussian(specimfwhm,dx)  ## REPLACED WITH SCIPY.SIGNAL.GAUSSIAN; UNSURE IF THIS IS CORRECT
        specim = specim/np.max(specim)

    #NOW GENERATE PIXEL FUNCTION
    #pixels are 18 microns
    pix = np.zeros(len(dx))
    inpix = np.where(abs(dx) <= 9.)[0]
    pix[inpix]=1.

    #NOW START THE CONVOLUTIONS
    #FIRST CONVOLVE SLIT WITH POINTILS
    ils0 = conv_circ(slit,pointils)#, mode='constant')

    #NEXT CONVOLVE THIS WITH SPECTROGRAPH IMAGE QUALITY
    if specimfwhm > 0: 
        ils1 = conv_circ(ils0,specim)#, mode='constant')
    else: 
        ils1 = ils0[:]

    #NEXT CONVOLVE THIS WITH PIXEL FUNCTION
    ils2 = conv_circ(ils1,pix)#, mode='constant')

    return wave,ils2


# if __name__ == '__main__':
#     inslit = {}
#     inslit['uniform'] = np.ones(slit_length)
#     inslit['point'] = np.zeros(slit_length)+0.01
#     inslit['point'][round(slit_length/5)] = 1.
#     inslit['half'] = np.zeros(slit_length)
#     inslit['half'][:round(slit_length/2)] = 1.
#     inslit['linear'] = np.linspace(0,1,slit_length)

#     ils = {}
#     for ky in names:
#         wave,tils = makeils4(band,inslit[ky])
#         ils[ky] = tils/(tils.sum()*np.diff(wave)[0])


#     fig = plt.figure(figsize=(12,8))
#     gs = GridSpec(3,1)
#     ax = fig.add_subplot(gs[0,0])
#     for ky in names:
#         ax.plot(inslit[ky])
#     #[x0,x1] = ax.get_xlim()
#     #[y0,y1] = ax.get_ylim()
#     #ax.set_aspect(0.4*(x1-x0)/(y1-y0))
#     plt.legend(ils.keys())
#     plt.title('Slit Functions')

#     ax = fig.add_subplot(gs[1,0])
#     inds = np.where(ils['uniform'] >= ils['uniform'].max()*0.002)[0]
#     for ky in names:
#         ax.plot(wave[inds],ils[ky][inds])#/ils.max())
#     #[x0,x1] = ax.get_xlim()
#     #[y0,y1] = ax.get_ylim()
#     #ax.set_aspect(0.4*(x1-x0)/(y1-y0))
#     plt.legend(ils.keys())
#     plt.title('Normalized ISRF for Different Slit Functions')

#     ax = fig.add_subplot(gs[2,0])
#     inds = np.where(ils['uniform'] >= ils['uniform'].max()*0.002)[0]
#     for ky in names:
#         ax.semilogy(wave[inds],ils[ky][inds])#/ils.max())
#     #[x0,x1] = ax.get_xlim()
#     #[y0,y1] = ax.get_ylim()
#     #ax.set_aspect(0.4*(x1-x0)/(log(y1)-log(y0)))
#     #plt.yscale('log')
#     plt.legend(ils.keys())
#     plt.title('Normalized ISRF for Different Slit Functions')
#     plt.tight_layout()

#     plt.savefig('slit_plot.png',bbox_inches='tight')

