#! /usr/bin/env python3


#*******************************************************************************
#
#*******************************************************************************
import argparse
import h5py
import numpy as np
import shutil


#*******************************************************************************
#
#*******************************************************************************
parser = argparse.ArgumentParser(description = 'For nerf hearding')

# positional arguments

parser.add_argument(dest = 'channel',
                    metavar = '<channel>',
                    type = int,
                    help = '')

parser.add_argument(dest = 'factor',
                    metavar = '<factor>',
                    type = float,
                    help = '')

parser.add_argument(dest = 'in_file',
                    metavar = '<filename>',
                    type = str,
                    help = '')

parser.add_argument(dest = 'out_file',
                    metavar = '<filename>',
                    type = str,
                    help = '')

# optional arguments

args = parser.parse_args()


#*******************************************************************************
#
#*******************************************************************************
shutil.copyfile(args.in_file, args.out_file)

f = h5py.File(args.out_file, 'r+')
if args.channel == 0:
    f['/SpectralConversion/ils_delta_lambda'][:,0,:,:] *= args.factor
else:
    f['/SpectralConversion/ils_delta_lambda'][args.channel - 1,0,:,:] *= args.factor
f.close()
