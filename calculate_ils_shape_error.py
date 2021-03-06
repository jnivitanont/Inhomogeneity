#!/usr/bin/env python3
# @author: Jeff Nivitanont, 2019-09-13

import h5py as h5
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculates shape error')
parser.add_argument('-i', '--ils', type=str, help='ILS file generated by makeils')
parser.add_argument('-u', '--unif', type=str, help='Uniform ILS file generated by makeils')
args = parser.parse_args()
ils_fid = args.ils
unif_fid = args.unif

def main():
    ils = h5.File(ils_fid, 'r+')
    unif = h5.File(unif_fid, 'r')
    instr_modes = ['ils_with_sh','ils_without_sh']
    for mode in instr_modes:
        scene_size = ils[mode].shape[1]
        unifils = unif[mode][:]
        unifmax = np.max(unifils,1)
        ils_se = np.full((4,scene_size), np.nan)
        for i in range(scene_size):
            ils_se[:,i] = (np.max(np.abs(ils[mode][:, i] - unifils),1).T/np.abs(unifmax)).T 
        try:
            del ils[mode + '_shape_error']
            print('Modifying', mode + '_shape_error')
        except:
            print('Creating', mode + '_shape_error')
        ils[mode + '_shape_error'] = ils_se
    ils.close()
    unif.close()
##END main

if __name__ == '__main__':
    main()