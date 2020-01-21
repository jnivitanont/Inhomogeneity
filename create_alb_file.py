#!/miniconda3/bin/python
## Author: Jeff Nivitanont
## Edited: 2019-03-22

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import h5py as h5
import sys
import shutil


## take in filenames from command line
scene_dir = str(sys.argv[1])
print('Scene file directory:\t'+scene_dir)
scene_filename = str(sys.argv[2])
print('File to be modified:\t'+scene_filename)
alb_dir = str(sys.argv[3])
print('Albedo directory:\t'+alb_dir)
lamont_filename = str(sys.argv[4])
print('Albedo file:\t'+lamont_filename)
mod_filename = 'mod_'+scene_filename
## create a copy of the HDF5 file to be modded
shutil.copyfile(scene_dir+scene_filename,scene_dir+ mod_filename)

scene = h5.File(scene_dir+mod_filename, 'r+')
lamont_bsa = xr.open_dataset(alb_dir + lamont_filename)

## Simple way to make albedos constant across each GeoCarb band
new_wave = np.repeat([[700, 800, 1500, 1700, 2000, 2100, 2500]], 2065, 0)
scene['Simulation/Surface/modis_wavelength'][:] = new_wave
scene['Simulation/Surface/modis_frequency'][:] =  1e7/new_wave
mod_alb = scene['Simulation/Surface/modis_albedo']
mod_alb[:,0] = lamont_bsa.Albedo_BSA_Band2.values
mod_alb[:,1] = lamont_bsa.Albedo_BSA_Band2.values
mod_alb[:,2] = lamont_bsa.Albedo_BSA_Band6.values
mod_alb[:,3] = lamont_bsa.Albedo_BSA_Band6.values
mod_alb[:,4] = lamont_bsa.Albedo_BSA_Band7.values
mod_alb[:,5] = lamont_bsa.Albedo_BSA_Band7.values
mod_alb[:,6] = lamont_bsa.Albedo_BSA_Band7.values
lamont_bsa.close()
scene.close()
print('Modified File:\t'+mod_filename)
