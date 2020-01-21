## Python // Plot albedo from high-res scene files
## Date: 2019-01-30
## Author: Jeff Nivitanont

import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
import datetime as dt
plt.switch_backend('agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import shapely.geometry as sgeom

plot_dir = '/home/jnivitanont/plots/'
scene_dir = '/home/jnivitanont/expo/lib/geocarb/dat_local/lamont/shin/100/output/'
retrieval_dir = '/home/jnivitanont/expo/lib/geocarb/dat_local/lamont/shin_01/100/retrieval/'
#scene_dir = 'C:/Users/Jeff/Desktop/Crowell-lab/Inhomogeneity/output/'
#retrieval_dir = scene_dir

#SUBSLITS_F = 275*11
#SLITS_C = 21*11

def find_missing_idx(idx, verbose=False):
    j=0
    missed = np.array([])
    for i in range(231):    # This is for a 21x11 footprint area
        if(idx[j] != i):
            if (verbose):
                print('Missing sounding index ' + str(i))
            missed = np.append(missed,i)
            j -=1
        j+=1
    return missed.astype(int)

def load_retrieval_var(retr_obj, var, **kwargs):
    result = retr_obj[var][:] #is flat array
    if (len(result) < 231):
        sounding_id = retr_obj['RetrievalHeader']['sounding_id_reference'][:]%1000
        missing = find_missing_idx(sounding_id, **kwargs)
        result = np.insert(result, (missing - 1), np.nan )
    result = np.reshape(result, (21,11))
    return result


def plot_lamont_var(var, varstr, res='low', cbar_lab=None):

    fig =  plt.figure(figsize=(10,10), dpi=300)
    ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m'),
                    facecolor='white' , edgecolor='black', linewidth=0.5,
                   zorder=0)
    ax1.gridlines(zorder=1, linewidth=0.2)
    ax1.axis('scaled')
    ax1.set_extent([-98, -97, 36.81, 36.4], crs=ccrs.PlateCarree())
    if res == 'high':
        mesh = ax1.pcolormesh( grid_lon_f, grid_lat_f, var, edgecolors='k', linewidths=.1)
    else:
        mesh = ax1.pcolormesh( grid_lon_c, grid_lat_c, var, edgecolors='k', linewidths=.1)
    ax1.add_patch( PolygonPatch(polygon=sgeom.Point(-97.486,36.604).buffer(.002), color='red',alpha=1, zorder=6))
    cbar = plt.colorbar(mesh,ax=ax1, shrink=.3)
    if cbar_lab is None:
        cbar_lab = varstr
    cbar.set_label(cbar_lab)
    plt.title(plot_dir + datetime.strftime('%Y%m%d%H%M, ') + varstr + ', ' + res + ' res' )
    plt.savefig(plot_dir + datetime.strftime('%Y%m%d%H%M_') + varstr + '_res' + res + '.png', bbox_inches='tight')
    return

scene_c = h5.File(scene_dir + 'lamont_shin_21x11.hdf', "r") #coarse
scene_f = h5.File(scene_dir + 'lamont_shin_275x11.hdf', "r") #fine
#scene_l1b = h5.File(output_dir + 'lamont_shin_21x11_l1b.hdf', "r")
retrieval = h5.File(retrieval_dir + 'lamont_shin_21x11.h5', 'r')

datetime = dt.datetime(*scene_f['Time/epoch'][0,])
ret_id = retrieval['RetrievalHeader/sounding_id_reference'][:]%1000
missing_id = find_missing_idx(ret_id)

## get coords for footprint corners
grid_lat_f = np.zeros([276,12])
grid_lat_f[0,0] = scene_f['Geometry/latitude_south_east'][0,0]
grid_lat_f[1:,:-1] = scene_f['Geometry/latitude_south_west'][:]
grid_lat_f[0,1:] = scene_f['Geometry/latitude_north_east'][0,:]
grid_lat_f[1:,-1] = scene_f['Geometry/latitude_north_west'][:,-1]

grid_lon_f = np.zeros([276,12])
grid_lon_f[0,0] = scene_f['Geometry/longitude_south_east'][0,0]
grid_lon_f[1:,:-1] = scene_f['Geometry/longitude_south_west'][:]
grid_lon_f[0,1:] = scene_f['Geometry/longitude_north_east'][0,:]
grid_lon_f[1:,-1] = scene_f['Geometry/longitude_north_west'][:,-1]
grid_lon_f = grid_lon_f - 360.

grid_lat_c = np.zeros([22,12])
grid_lat_c[0,0] = scene_c['Geometry/latitude_south_east'][0,0]
grid_lat_c[1:,:-1] = scene_c['Geometry/latitude_south_west'][:]
grid_lat_c[0,1:] = scene_c['Geometry/latitude_north_east'][0,:]
grid_lat_c[1:,-1] = scene_c['Geometry/latitude_north_west'][:,-1]

grid_lon_c = np.zeros([22,12])
grid_lon_c[0,0] = scene_c['Geometry/longitude_south_east'][0,0]
grid_lon_c[1:,:-1] = scene_c['Geometry/longitude_south_west'][:]
grid_lon_c[0,1:] = scene_c['Geometry/longitude_north_east'][0,:]
grid_lon_c[1:,-1] = scene_c['Geometry/longitude_north_west'][:,-1]
grid_lon_c = grid_lon_c - 360.


#plot SNRs for all 4 bands
snr = np.zeros((21,11,4))
for band in enumerate(['o2', 'weak_co2', 'strong_co2', 'ch4']):
    snr_temp = retrieval['SpectralParameters/signal_' + band[1] + '_fph'][:]/retrieval['SpectralParameters/noise_' + band[1] +'_fph'][:]
    snr_temp = np.insert(snr_temp,np.array(missing_id)-1, np.nan )
    snr[:,:,band[0]] = np.reshape(snr_temp, (21,11))
    print('SNR ' + band[1] + 'band')
    print(snr[:,:,band[0]])
    plot_lamont_var(var=snr[:,:,band[0]], varstr=('SNR_' + band[1]), res='low')

#plot albedos for all 4 bands
albedo_f = scene_f['Surface/modis_albedo'][:,:,(1,5,6)]
albedo_c = scene_c['Surface/modis_albedo'][:,:,(1,5,6)]
modis_bands = np.array([2,6,7])
for bnd in range(3):
    plot_lamont_var(var=albedo_f[:,:,bnd], varstr=('albedo_band'+ str(modis_bands[bnd])), res='high' )
for bnd in range(3):
    plot_lamont_var(var=albedo_c[:,:,bnd], varstr=('albedo_band'+ str(modis_bands[bnd])))

#plot total error xco2
scene_xco2 = np.mean(scene_c['Gas']['volume_mixing_ratio'][:,:,:,3], axis=2)*1e6
retr_xco2  = load_retrieval_var(retrieval,'RetrievalResults/xco2')*1e6
xco2_bias = retr_xco2 - scene_xco2
plot_lamont_var(xco2_bias, varstr='xco2_bias')
print('xco2 bias')
print(xco2_bias)

## plot other vars
for retrvar in ['xco2_uncert']:
    tempvar = load_retrieval_var(retrieval,'RetrievalResults/'+retrvar)*1e6
    plot_lamont_var(tempvar, varstr=retrvar)
    print(retrvar)
    print(tempvar)

