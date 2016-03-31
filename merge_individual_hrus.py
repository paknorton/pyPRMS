#!/usr/bin/env python

import os
import glob
import sys

import prms_lib as prms

__version__ = '0.1'

mod_params = ['tmax_allsnow', 'tmax_allrain', 'rad_trncf', 'freeh2o_cap', 
              'emis_noppt', 'cecn_coef', 'potet_sublim']

region = 'r10U'

base_dir = '/Volumes/data/Users/pnorton/USGS/Projects/National_Hydrology_Model'
parent_dir = '%s/regions/%s' % (base_dir, region)
src_dir = '/Volumes/LaCie/20150720a/regions/%s_byHRU' % region
dst_dir = '/Volumes/LaCie/20150720a/regions/%s_byHRU' % region

control_file = '%s/control/daymet.control' % parent_dir

# Read the control file
control = prms.control(control_file)

# Read the input parameter file
orig_params = prms.parameters('%s/%s' % (parent_dir, control.get_var('param_file')['values'][0]))

param_filename = os.path.basename(control.get_var('param_file')['values'][0])

# TODO: Right now only nhru is handled for checking in changed parameters; this should be expanded for nssr and ngw
# split_dims = set(['nhru', 'nssr', 'ngw'])

print('%s/%s_*/%s' % (src_dir, region, param_filename))
proc_list = [el for el in glob.glob('%s/%s_*/%s' % (src_dir, region, param_filename))]

for dd in proc_list:
    # Read the parameter file for the current HRU and get it's parent hru index
    new_params = prms.parameters(dd)
    parent_hru = new_params.get_var('parent_hru')['values'][0]

    sys.stdout.write('\r\tHRU: %06d ' % parent_hru)
    sys.stdout.flush()

    # Only check-in the the parameters that were changed
    # Currently this is read from merged_params.txt
    for ii in mod_params:
        curr_param = new_params.get_var(ii)
        orig_params.update_values_by_hru(ii, curr_param['values'], parent_hru)
print('')

orig_params.write_param_file('crap_new_param')
