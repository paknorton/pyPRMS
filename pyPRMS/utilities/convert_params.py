#!/usr/bin/env python

import argparse
import os

from pyPRMS import ParamDb
from pyPRMS.metadata.metadata import MetaData
from pyPRMS import ParameterFile

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Convert parameter files to different formats')
    parser.add_argument('--src', help='Source file or directory')
    parser.add_argument('--fmt', help='Output format', choices=['paramdb', 'classic', 'netcdf'],
                        default='paramdb')
    parser.add_argument('--dst', help='Destination file or directory')

    args = parser.parse_args()

    # Check and read the source
    print('Reading source parameters')

    prms_meta = MetaData(verbose=False).metadata

    if os.path.isdir(args.src):
        # If a directory is provided for the source we assume it is a
        pdb = ParamDb(args.src, metadata=prms_meta)
    elif os.path.isfile(args.src):
        # A parameter file in either classic format or netcdf format
        if os.path.splitext(args.src)[1] == '.param':
            # Classic parameter file
            pdb = ParameterFile(args.src, metadata=prms_meta, verbose=False)
        else:
            print('Only parameter databases and classic parameter files are currently supported for source files')
            exit()
    else:
        print('Source argument is neither a file or directory')
        exit()

    # Try to write the file out
    if args.fmt == 'netcdf':
        print('Writing parameters to netcdf format')
        pdb.write_netcdf(args.dst)
    elif args.fmt == 'paramdb':
        print('Writing parameters to paramdb format')
        pdb.write_paramdb(args.dst)
    elif args.fmt == 'classic':
        print('Writing parameters to parameter file')
        header = ['Written by pyPRMS.convert_params']
        pdb.write_parameter_file(args.dst, header=header)
    else:
        print('Currently only netcdf, paramdb, or parameter file output is supported.')
        exit()
    print('Done.')


if __name__ == '__main__':
    main()
