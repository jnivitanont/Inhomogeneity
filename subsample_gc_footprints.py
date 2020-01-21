from netCDF4 import Dataset
from pylab import *
from h5py import File
import subprocess as sbp
import pdb,sys
import argparse

parser = argparse.ArgumentParser(description='Subsample GC footprints from MODIS files')
parser.add_argument('-l','--location', metavar='location', required=True, help='Target location; "lamont" or "manaus"')
parser.add_argument('-d','--date', metavar='date', required=True, help='MODIS file date')
parser.add_argument('-a','--arp', metavar='ARP file', required=True, help='ARP file')
args = parser.parse_args()

site = args.location#sys.argv[1] #'lamont' or 'manaus'
date = args.date#sys.argv[2] #YYYY.mm.dd
arp_fid = args.arp
geom_fid = '/data10/jnivitanont/ils_calcs_files/geometry_'+site+'_2x2.hdf'
#arp_fid = '/data10/jnivitanont/csu_sim/data/ARP/geocarb/geocarb_sim_ARP_20180305_4band_1foot_fixed_new_snr.h5'
if site == 'lamont':
    cont = 'north_america_'
elif site == 'manaus':
    cont = 'south_america_'
modis_fid = '/data10/jnivitanont/ils_calcs_files/'+cont+date+'.nc4'
fid_hr = Dataset(modis_fid,'r')
lat = fid_hr['latitude'][:]
lon = fid_hr['longitude'][:]
bands = {}
bands['band7'] = fid_hr['Albedo_BSA_Band7'][:]
bands['band6'] = fid_hr['Albedo_BSA_Band6'][:]
bands['band2'] = fid_hr['Albedo_BSA_Band2'][:]

fid_gc = File(geom_fid, 'r')
ils_grid = File(arp_fid, 'r')['SpectralConversion/ils_delta_lambda'][:]

lat_ne = fid_gc['Geometry/latitude_north_east'][:].flatten()
lon_ne = fid_gc['Geometry/longitude_north_east'][:].flatten()-360.
lat_nw = fid_gc['Geometry/latitude_north_west'][:].flatten()
lon_nw = fid_gc['Geometry/longitude_north_west'][:].flatten()-360.
lat_se = fid_gc['Geometry/latitude_south_east'][:].flatten()
lon_se = fid_gc['Geometry/longitude_south_east'][:].flatten()-360.
lat_sw = fid_gc['Geometry/latitude_south_west'][:].flatten()
lon_sw = fid_gc['Geometry/longitude_south_west'][:].flatten()-360.
lat_g = fid_gc['Geometry/latitude_centre'][:].flatten()
lon_g = fid_gc['Geometry/longitude_centre'][:].flatten()-360.
lamont_lat = 36.6
lamont_lon = -97.5

g_loc_inds = range(lat_ne.shape[0])#where((lat_g > lamont_lat-1)*(lat_g < lamont_lat+1)*(lon_g > lamont_lon-1)*(lon_g < lamont_lon+1))
glat_ne_loc = lat_ne[g_loc_inds]
glon_ne_loc = lon_ne[g_loc_inds]
glat_nw_loc = lat_nw[g_loc_inds]
glon_nw_loc = lon_nw[g_loc_inds]
glat_se_loc = lat_se[g_loc_inds]
glon_se_loc = lon_se[g_loc_inds]
glat_sw_loc = lat_sw[g_loc_inds]
glon_sw_loc = lon_sw[g_loc_inds]
glat_loc = lat_g[g_loc_inds]
glon_loc = lon_g[g_loc_inds]
galb_loc = zeros(glon_ne_loc.shape)
#dlat = abs(diff(lat_g,axis=0))
#dlon = abs(diff(lon_g,axis=0))
#dlat_loc = 0.035*ones(glat_loc.shape)#dlat[g_loc_inds] %avg using dlat = 4.*180./6400/pi
#dlon_loc = 0.07*ones(glon_loc.shape)#dlon[g_loc_inds] %avg using simple formula dlon = 6.5/cos(lat_g*pi/180.)/60./1.9

loc = {}
for il in range(len(glat_loc)):
    loc[il] = {}
#    lat_inds = where(abs(lat-glat_loc[il]) < dlat_loc[il])[0]
#    lon_inds = where(abs(lon-glon_loc[il]) < dlon_loc[il])[0]
    lat_inds = where((lat >= glat_se_loc[il])*(lat >= glat_sw_loc[il])*(lat <= glat_ne_loc[il])*(lat <= glat_nw_loc[il]))[0]
    lon_inds = where((lon <= glon_se_loc[il])*(lon >= glon_sw_loc[il])*(lon <= glon_ne_loc[il])*(lon >= glon_nw_loc[il]))[0]
    lat_hr = lat[lat_inds]
    lon_hr = lon[lon_inds]
    alb_hr = {}
    lon_hr_g = linspace(lon_hr.min(),lon_hr.max(),20)
    for b in bands.keys():
        loc[il]['alb_hr_'+b] = array([[bands[b][jlat,jlon] for jlon in lon_inds] for jlat in lat_inds])
        alb_hr_eq = nanmean(loc[il]['alb_hr_'+b],axis=0)
        n_nans = sum(np.isnan(alb_hr_eq))
        if n_nans > 0:
            nan_inds = np.where(np.isnan(alb_hr_eq))[0]
            x = np.arange(len(alb_hr_eq))
            good_inds = np.where(1-np.isnan(alb_hr_eq))[0]
            if len(good_inds) > 2:
                f = np.poly1d(polyfit(x[good_inds],alb_hr_eq[good_inds],1))
                alb_hr_eq[nan_inds] = f(nan_inds)
            #pdb.set_trace()
            #for ind in nan_inds:
            #    lb_ind = np.where(1-np.isnan(alb_hr_eq[nan_inds[0]-1:ind]))[0][-1]
            #    ub_ind = np.where(1-np.isnan(alb_hr_eq[ind:nan_inds[-1]+2]))[0][0]
            #    alb_hr_eq[ind] = alb_hr_eq[lb_ind] + (alb_hr_eq[ub_ind]-alb_hr_eq[lb_ind])/float(ub_ind-lb_ind)*(ind-lb_ind)
        loc[il]['alb_hr_g_'+b] = interp(lon_hr_g,lon_hr,alb_hr_eq)
        loc[il]['gc_fp_mean_alb_'+b] = nanmean(array([[bands[b][jlat,jlon] for jlon in lon_inds] for jlat in lat_inds]))
        loc[il]['gc_fp_cv_alb_'+b] = nanstd(loc[il]['alb_hr_'+b])/nanmean(loc[il]['alb_hr_'+b])
    loc[il]['lon_hr_g'] = lon_hr_g[:]
    loc[il]['lat_hr'] = lat[lat_inds]
    loc[il]['lon_hr'] = lon[lon_inds]
    loc[il]['gc_lat'] = glat_loc[il]
    loc[il]['gc_lon'] = glon_loc[il]

fid_out = Dataset('/data10/jnivitanont/ils_calcs_files/albedo_variation/'+site+'_ew_albedo_variation_'+date+'.nc4','w')
fid_out.createDimension('n_footprints',len(glat_loc))
fid_out.createDimension('slit_points',len(lon_hr_g))
fid_out.createDimension('ils_points',ils_grid.shape[-1])
fid_out.createDimension('n_bands',4)

fid_out.createVariable('gc_footprint_lat','f8','n_footprints')
fid_out['gc_footprint_lat'][:] = array([loc[il]['gc_lat'] for il in sorted(loc.keys())])
fid_out.createVariable('gc_footprint_lon','f8','n_footprints')
fid_out['gc_footprint_lon'][:] = array([loc[il]['gc_lon'] for il in sorted(loc.keys())])
for b in bands.keys():
    fid_out.createVariable('gc_footprint_alb_'+b,'f8','n_footprints')
    fid_out['gc_footprint_alb_'+b][:] = array([loc[il]['gc_fp_mean_alb_'+b] for il in sorted(loc.keys())])
    fid_out.createVariable('gc_footprint_alb_cv_'+b,'f8','n_footprints')
    fid_out['gc_footprint_alb_cv_'+b][:] = array([loc[il]['gc_fp_cv_alb_'+b] for il in sorted(loc.keys())])

fid_out.createVariable('gc_subslit_lon','f8',('n_footprints','slit_points'))
fid_out['gc_subslit_lon'][:] = array([loc[il]['lon_hr_g'][:] for il in sorted(loc.keys())])
for b in bands.keys():
    fid_out.createVariable('gc_subslit_alb_'+b,'f8',('n_footprints','slit_points'))
    fid_out['gc_subslit_alb_'+b][:] = array([loc[il]['alb_hr_g_'+b] for il in sorted(loc.keys())])
fid_out.createVariable('ils_delta_lambda','f8',('n_bands','ils_points'))
fid_out['ils_delta_lambda'][:] = ils_grid[:,0,0,:]
fid_out.createVariable('ils_with_sh','f8',('n_bands','n_footprints','ils_points'))
fid_out['ils_with_sh'][:] = 0.
fid_out.createVariable('ils_without_sh','f8',('n_bands','n_footprints','ils_points'))
fid_out['ils_without_sh'][:] = 0.
fid_out.close()




