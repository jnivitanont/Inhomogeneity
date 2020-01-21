#!/usr/bin/env python3
import h5py as h5
import argparse
import shutil

parser = argparse.ArgumentParser(description='Merge footprint L1B files.')
parser.add_argument('-r', '--root', metavar='root', required=True, help='Footprint filename root')
parser.add_argument('-u', '--uniform', metavar='uniform', required=True, help='L1B file to modify')
parser.add_argument('-l', '--location', metavar='location', required=True, help='Lamont or Manaus')
args = parser.parse_args()

n_footprints = {'lamont':2065, 'manaus':2880}
l1b_root = args.root
unif_fid = args.uniform
location = args.location
mod_fid = l1b_root + 'foot_merged.hdf'
groups = ['SoundingMeasurements', 'SpikeEOF']
missing_footprints = list()

if __name__ == '__main__':
    print(args)
    shutil.copyfile(unif_fid, mod_fid)
    try:
        mod_file = h5.File(mod_fid, 'r+')
    except:
        print('Modify file not found.')
        raise

    print('Copying Footprints:')
    for i in range(n_footprints[location]):
        if i%300 == 0:
            print("..",str(i).zfill(4), end="" )

        try:
            temp = h5.File(l1b_root + str(i).zfill(4) + '.hdf', 'r')
            for grp in groups:
                key = grp
                datasets = list(temp[key])
                for dat in datasets:
                    key = grp + '/' + dat
                    mod_file[key][[i]] = temp[key][[0]]
            temp.close()
        except:
            print('Footprint ' + str(i).zfill(4) + ' not found.')
            #raise
            missing_footprints.append(str(i).zfill(4))
    mod_file.close()
    print('L1b merging complete.')
    print('Missing footprints:',missing_footprints)
