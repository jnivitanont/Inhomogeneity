#!/usr/bin/env python
## This scripts creates individual ARP files for each footprint
## by Jeff Nivitanont, GeoCarb (U. Okla.). 2019/07/22

import netCDF4 as nc
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import argparse

parser = argparse.ArgumentParser(description='Split ARP files for individual footprints.')
parser.add_argument('-l','--location', metavar='location', required=True, help='Target location; "lamont" or "manaus".')
parser.add_argument('-m','--month', metavar='month', required=True, help='Scan month.')
parser.add_argument('-i','--ils', metavar='ils file', required=True, help='File containing ILS.')
parser.add_argument('-a','--arp', metavar='arp file', required=True, help='ARP file.')
args = parser.parse_args()

location = args.location 
month = args.month
ils_fid = args.ils
arp_fid = args.arp

arp_dir, arp_filename = arp_fid.rsplit('/',1)
file_dir = arp_dir + '/mod/' + location + '/2016'+month+'21/'
scene_size = {'lamont':2065, 'manaus':2880}
#file_dir = '/data10/jnivitanont/csu_sim/data/ARP/geocarb/mod/' + location + '/2016' + month + '21/'
#ils_fid  = '/data10/jnivitanont/ils_calcs_files/albedo_variation/' + location + '_ew_albedo_variation_2014.' + month + '.18.nc4'
#arp_fid  = '/data10/jnivitanont/csu_sim/data/ARP/geocarb/geocarb_sim_ARP_20180305_4band_1foot_fixed_new_snr.h5'

#source_arp = h5.File(arp_fid,'r')
if __name__ == '__main__':
    ils = nc.Dataset(ils_fid, 'r')
    for ils_config in ['ils_with_sh', 'ils_without_sh']:
        print('ILS configuration: ' + ils_config)
        file_dest = file_dir + ils_config
        if not os.path.exists(file_dest):
            os.makedirs(file_dest)
        for j in range(scene_size[location]):  #footprint
            mod_fid  = file_dest + '/geocarb_sim_ARP_20180305_4band_1foot_fixed_new_snr_' + \
                            str(j).zfill(4) + '.h5'
            if os.path.exists(mod_fid):
                        print('modifying ' + mod_fid)
            else:
                        print('creating ' + mod_fid)
                        shutil.copyfile(arp_fid, mod_fid)
            arp = h5.File(mod_fid, 'r+')
            for i in range(4):  #band
                temp = ils[ils_config][i,j,:]
                grid = ils['ils_delta_lambda'][i,:]
        #        fwhm = np.abs(grid[temp> temp.max()/2]).max()*2
                arp['SpectralConversion/ils_delta_lambda'][i,0,:,:] = np.tile(grid, (947,1))
                arp['SpectralConversion/ils_relative_response'][i,0,:,:] = np.tile(temp, (947,1))
            arp.close()
        print('ARP modding complete.')
        print('ARP files saved to ' + file_dest)
    ils.close()