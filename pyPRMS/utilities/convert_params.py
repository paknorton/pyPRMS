#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

import argparse
import os
# import shutil

from pyPRMS.NhmParamDb import NhmParamDb
from pyPRMS.ParameterFile import ParameterFile

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Convert parameter files to different formats')
    parser.add_argument('--src', help='Source file or directory')
    parser.add_argument('--dst', help='Destination file or directory')

    args = parser.parse_args()

    output_netcdf = False
    output_classic = False
    output_paramdb = False

    # Check the destination
    # print('dst path:', os.path.splitext(args.dst)[0])
    if os.path.splitext(args.dst)[1] == '.nc':
        print('- output to netcdf')
        output_netcdf = True
    elif os.path.splitext(args.dst)[1] == '.param':
        print('- output to parameter file')
        output_classic = True
    elif os.path.basename(args.dst) == 'paramdb':
        print('- output to paramdb')
        output_paramdb = True
    else:
        print('Currently only netcdf, paramdb, or parameter file output is supported.')
        exit()

    # Check and read the source
    print('Reading source parameters')
    if os.path.isdir(args.src):
        # If a directory is provided for the source we assume it is a
        # paramdb format.
        params = NhmParamDb(args.src)
    elif os.path.isfile(args.src):
        # A parameter file in either classic format or netcdf format
        if os.path.splitext(args.src)[1] == '.param':
            # Classic parameter file
            params = ParameterFile(args.src)
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
    elif output_classic:
        print('Writing parameters to parameter file')
        header = ['Written by pyPRMS.convert_params']
        params.write_parameter_file(args.dst, header=header)
    print('Done.')


if __name__ == '__main__':
    main()