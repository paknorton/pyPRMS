#!/usr/bin/env python

from __future__ import print_function
# from builtins import range
from six import iteritems

import os
import pandas as pd
import subprocess
import sys
import argparse

import prms_lib as prms

__version__ = '0.3'

# region = 'r10U'
#
# base_dir = '/Volumes/data/Users/pnorton/USGS/Projects/National_Hydrology_Model'
# src_dir = '%s/regions/%s' % (base_dir, region)
# dst_dir = '/Volumes/LaCie/20150720a/regions/%s_byHRU' % region
#
# dummy_streamflow_file = '%s/regions/1960-2010.data.dummy' % (base_dir)
# control_file = '%s/control/daymet.control' % (src_dir)

cbh_vars = {'tmax_day': 'tmax', 'tmin_day': 'tmin', 'precip_day': 'prcp'}
input_files = ['param_file', 'tmax_day', 'tmin_day', 'precip_day', 'data_file']
output_files = ['ani_output_file', 'var_init_file', 'stat_var_file', 'var_save_file', 'stats_output_file',
                'csv_output_file', 'model_output_file']

control_updates = {'statVar_names': ['hru_streamflow_out', 'pkwater_equiv', 'hru_actet', 'hru_ppt'],
                   'statVar_element': [1, 1, 1, 1],
                   'statsON_OFF': [1]}


def cbh_subset(infile, outfile, varname, hruindex):
    # Create a *.cbh file with a single HRU
    valid_varnames = ['prcp', 'tmin', 'tmax']
    missing = [-99.0, -999.0]
    
    in_hdl = open(infile, 'r')
    
    # Read the header information
    fheader = ''
    
    for ii in range(0, 3):
        line = in_hdl.readline()
        
        if line[0:4] in valid_varnames:
            # Increment number of HRUs that are included
            numhru = int(line[5:])
            fheader += line[0:5] + ' 1\n'
        else:
            fheader += line
                         
    # Read the data
    df1 = pd.read_csv(in_hdl, sep=' ', skipinitialspace=True,
                      usecols=[0, 1, 2, 3, 4, 5, hruindex+5],
                      header=None)
    in_hdl.close()
    
    # Write the HRU subset of data out to a file
    out_hdl = open(outfile, 'w')
    out_hdl.write(fheader)
    
    df1.to_csv(out_hdl, sep=' ', float_format='%0.4f', header=False, index=False)
    out_hdl.close()
    

def full_cbh_subset(src_file, dst_dir, region, varname, nhru, hdf=False):
    valid_varnames = ['prcp', 'tmin', 'tmax']
    missing = [-99.0, -999.0]

    fheader = ''

    if hdf:
    # if os.path.splitext(src_file)[1] == '.h5':
        # HDF5 file given
        df1 = pd.read_hdf('%s.h5' % os.path.splitext(src_file)[0], key=varname)
        df1['year'] = df1.index.year
        df1['month'] = df1.index.month
        df1['day'] = df1.index.day
        df1['hour'] = 0
        df1['minute'] = 0
        df1['second'] = 0
        df1.reset_index(drop=True, inplace=True)

        fheader = '{0:s} 1\n####\n'.format(varname)
    else:
        in_hdl = open(src_file, 'r')

        # Read the header information
        for ii in range(0, 3):
            line = in_hdl.readline()

            if line[0:4] in valid_varnames:
                # Increment number of HRUs that are included
                # numhru = int(line[5:])
                fheader += line[0:5] + ' 1\n'
            else:
                fheader += line

        # Read the data
        df1 = pd.read_csv(in_hdl, sep=' ', skipinitialspace=True, engine='c', header=None)
        in_hdl.close()

    for hh in range(nhru):
        sys.stdout.write('\r\t\t{0:06d} '.format(hh + 1))
        sys.stdout.flush()

        # Write the HRU subset of data out to a file
        out_hdl = open('{0:s}/r{1:s}_{2:06d}/{3:s}'.format(dst_dir, region, hh, os.path.basename(src_file)), 'w')
        out_hdl.write(fheader)
        
        # Subset dataframe to current HRU
        if hdf:
            # if os.path.splitext(src_file)[1] == '.h5':
            df1_ss = df1.ix[:, ['year', 'month', 'day', 'hour', 'minute', 'second', (hh+1)]]
        else:
            df1_ss = df1.iloc[:, [0, 1, 2, 3, 4, 5, hh+6]]
        
        df1_ss.to_csv(out_hdl, sep=' ', float_format='%0.4f', header=False, index=False)
        out_hdl.close()
    print('\n')


def main():
    parser = argparse.ArgumentParser(description='Split PRMS model into individual HRUs')
    parser.add_argument('-b', '--basedir', help='Base directory', required=True)
    parser.add_argument('-c', '--control', help='Name of control file', required=True)
    parser.add_argument('-d', '--dummy', help='Name of dummy streamflow datafile', required=True)
    parser.add_argument('-r', '--region', help='Region to process', required=True)
    parser.add_argument('--hdf', help='Read hdf5 CBH files', action='store_true')
    parser.add_argument('--nocbh', help='Skip CBH file splitting', action='store_true')

    args = parser.parse_args()

    region = args.region
    base_dir = args.basedir
    src_dir = '{0:s}/r{1:s}'.format(base_dir, region)
    dst_dir = '{0:s}/r{1:s}_byHRU'.format(base_dir, region)
    dummy_streamflow_file = '{0:s}/{1:s}'.format(base_dir, args.dummy)
    control_file = '{0:s}/r{1:s}/{2:s}'.format(base_dir, region, args.control)

    # Read the control file
    control = prms.control(control_file)

    # Get the input filename/paths
    input_src = {}
    for ii in input_files:
        input_src[ii] = control.get_var(ii)['values'][0]
        
    # Get the output filename/paths
    output_src = {}
    for ii in output_files:
        output_src[ii] = control.get_var(ii)['values'][0]

    # Update specified control variables
    for kk, vv in iteritems(control_updates):
        control.replace_values(kk, vv)
        
    # Modify the input and output file entries to remove the path
    for kk, vv in iteritems(input_src):
        if kk == 'data_file':
            newfilename = os.path.basename(dummy_streamflow_file)
        else:
            newfilename = os.path.basename(vv)
            
        control.replace_values(kk, newfilename)
        
    for kk, vv in iteritems(output_src):
        control.replace_values(kk, os.path.basename(vv))

    # Read the input parameter file
    params = prms.parameters('{0:s}/{1:s}'.format(src_dir, input_src['param_file']))

    nhru = params.get_dim('nhru')
    print('Number of HRUs to process:', nhru)

    for hh in range(nhru):
        # set and create destination directory
        cdir = '{0:s}/r{1:s}_{2:06d}'.format(dst_dir, region, hh)
        
        # Create the destination directory
        try:
            os.makedirs(cdir)
        except OSError:
            # One option if the directory already exists is to erase it
            # and re-create. For now we'll just pass
            pass
            # Directory already exists, delete it and re-create
            # shutil.rmtree(cdir)
            # os.makedirs(cdir)
        
        # Write an input parameter file for the current HRU 
        params.pull_hru2(hh, '{0:s}/{1:s}'.format(cdir, os.path.basename(input_src['param_file'])))
        
        # Write the control file for the current HRU
        control.write_control_file('{0:s}/{1:s}'.format(cdir, os.path.basename(control_file)))

        # Copy dummy streamflow data file
        cmd_opts = ' {0:s} {1:s}/.'.format(dummy_streamflow_file, cdir)
        subprocess.call('/bin/cp' + cmd_opts, shell=True)

    if not args.nocbh:
        # Write *.cbh subsets
        print('\nCBH subsets')
        for kk, vv in iteritems(cbh_vars):
            if kk in input_src:
                print('\tWriting {0:s}'.format(os.path.basename(input_src[kk])))

                if args.hdf:
                    full_cbh_subset('{0:s}/{1:s}'.format(src_dir, input_src[kk]), dst_dir, region, vv, nhru, hdf=True)
                else:
                    full_cbh_subset('{0:s}/{1:s}'.format(src_dir, input_src[kk]), dst_dir, region, vv, nhru)


if __name__ == '__main__':
    main()
