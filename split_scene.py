#!/usr/bin/env python
## Author: Jeff Nivitanont
## 2019-09-18

import os
import h5py as h5
import argparse

parser = argparse.ArgumentParser(description='Split footprint scene files.')
parser.add_argument('-s', '--scene', metavar='scene file ', required=True, help='Scene file')
args = parser.parse_args()
scene_fid = args.scene

if __name__ == '__main__':
    scene_dir, scene_filename = scene_fid.rsplit('/',1) #separate name from dir
    assert scene_filename.endswith('.hdf'), 'Not an HDF file' #check HDF file
    foot_filename = scene_filename[:-4] #remove .hdf
    if foot_filename.endswith('.hdf'):
        foot_filename = foot_filename[:-4]
    #read-in file
    scene = h5.File(scene_fid, 'r')
    scene_size = scene['Simulation/Geometry/latitude'].shape[0]
    #make new dir
    splice_dir = scene_dir + '/split/'
    if not os.path.exists(splice_dir):
        os.makedirs(splice_dir)
    groups = list(scene['Simulation'])
    groups.remove('Metadata')
    # split scene into individual footprint files
    for i in range(scene_size):
        temp_fid = splice_dir + foot_filename + '_' + str(i).zfill(4) + '.hdf'
        print('Creating', temp_fid)
        if os.path.exists(temp_fid):
            os.remove(temp_fid)
        temp = h5.File(temp_fid, 'a')
        for grp in groups:
            key = '/Simulation/' + grp
            datasets = list(scene[key])
            for dat in datasets:
                key = '/Simulation/' + grp + '/' + dat
                temp[key] = scene[key][[i]]
        key = '/Simulation/Metadata'
        for dat in list(scene[key]):
            key = '/Simulation/Metadata/' + dat
            temp[key] = scene[key][:]
        temp.close()
