#!/usr/bin/env python
## Author: Jeff Nivitanont
## Edited: 2019-09-16

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import shutil
import argparse

parser = argparse.ArgumentParser(description='Creates a MODIS albedo modified scene file.')
parser.add_argument('-s', '--scene', metavar='scene file', required=True, help='Scene File')
parser.add_argument('-a', '--albedo', metavar='albedo file', required=True,  help='Albedo File')
args = parser.parse_args()
scene_fid = args.scene
alb_fid = args.albedo
scene_dir, scene_filename = scene_fid.rsplit('/',1) #separate name from dir
mod_fid = scene_dir + '/mod_' + scene_filename
if __name__ == '__main__':
    ## create a copy of the HDF5 file to be modded
    shutil.copyfile(scene_fid, mod_fid)
    scene = h5.File(mod_fid, 'a')
    albedo = h5.File(alb_fid, 'r')
    scene_size = scene['Simulation/Surface/modis_albedo'][:].shape[0]
    ## Simple way to make albedos constant across each GeoCarb band
    new_wave = np.repeat([[700, 800, 1500, 1700, 2000, 2100, 2500]], scene_size, 0)
    scene['Simulation/Surface/modis_wavelength'][:] = new_wave
    scene['Simulation/Surface/modis_frequency'][:] =  1e7/new_wave
    scene['Simulation/Surface/modis_albedo'][:] = 0.0
    scene['Simulation/Surface/modis_brdf'][:,1] = 0.0
    scene['Simulation/Surface/modis_brdf'][:,2] = 0.0
    mod_brdf = scene['Simulation/Surface/modis_brdf']
    mod_alb = scene['Simulation/Surface/modis_albedo']
    mod_brdf[:,0,0] = albedo['gc_footprint_alb_band2'][:scene_size]*1e-3 
    mod_brdf[:,0,1] = albedo['gc_footprint_alb_band2'][:scene_size]*1e-3
    mod_brdf[:,0,2] = albedo['gc_footprint_alb_band6'][:scene_size]*1e-3
    mod_brdf[:,0,3] = albedo['gc_footprint_alb_band6'][:scene_size]*1e-3
    mod_brdf[:,0,4] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    mod_brdf[:,0,5] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    mod_brdf[:,0,6] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    mod_alb[:,0] = albedo['gc_footprint_alb_band2'][:scene_size]*1e-3
    mod_alb[:,1] = albedo['gc_footprint_alb_band2'][:scene_size]*1e-3
    mod_alb[:,2] = albedo['gc_footprint_alb_band6'][:scene_size]*1e-3
    mod_alb[:,3] = albedo['gc_footprint_alb_band6'][:scene_size]*1e-3
    mod_alb[:,4] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    mod_alb[:,5] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    mod_alb[:,6] = albedo['gc_footprint_alb_band7'][:scene_size]*1e-3
    albedo.close()
    scene.close()
    print('Modified File:\t'+mod_fid)
