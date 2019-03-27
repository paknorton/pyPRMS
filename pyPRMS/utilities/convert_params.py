#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import argparse
import os
# import shutil

import pyPRMS.NhmParamDb as pdb
import pyPRMS.ParameterFile as pfile

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Convert parameter files to different formats')
    parser.add_argument('--src', help='Source file or directory')
    parser.add_argument('--dst', help='Destination file or directory')

    args = parser.parse_args()

    output_netcdf = False
    output_classic = False

    # Check the destination
    # print('dst path:', os.path.splitext(args.dst)[0])
    if os.path.splitext(args.dst)[1] == '.nc':
        print('- output to netcdf')
        output_netcdf = True
    elif os.path.splitext(args.dst)[1] == '.param':
        print('Output to parameter file format is not supported yet.')
        exit()
    elif os.path.basename(args.dst) == 'paramdb':
        print('- output to paramdb')
        output_paramdb = True
    else:
        print('Currently only netcdf or paramdb output is supported.')
        exit()

    # Check and read the source
    print('Reading source parameters')
    if os.path.isdir(args.src):
        # If a directory is provided for the source we assume it is a
        # paramdb format.
        params = pdb.NhmParamDb(args.src)
    elif os.path.isfile(args.src):
        # A parameter file in either classic format or netcdf format
        if os.path.splitext(args.src)[1] == '.param':
            # Classic parameter file
            params = pfile.ParameterFile(args.src)
        else:
            print('Only classic parameter files are currently supported for source files')
            exit()
    else:
        print('Source argument is neither a file or directory')
        exit()

    # Try to write the file out
    if output_netcdf:
        print('Writing parameters to netcdf format')
        params.write_netcdf(args.dst)
    elif output_paramdb:
        print('Writing parameters to paramdb format')
        params.write_paramdb(args.dst)
    print('Done.')


if __name__ == '__main__':
    main()
