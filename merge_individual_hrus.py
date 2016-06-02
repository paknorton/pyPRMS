#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
# from future.utils import iteritems


import os
import glob
import sys
import numpy as np

import prms_lib as prms

__version__ = '0.1'

# TODO: Get parameters to merge from the calibration
mod_params = ['tmax_allsnow', 'rad_trncf', 'freeh2o_cap',
              'emis_noppt', 'cecn_coef', 'potet_sublim', 'tmin_cbh_adj']

region = 'rTest'

base_dir = '/Users/pnorton/Projects/National_Hydrology_Model'
parent_dir = '{0:s}/regions/{1:s}'.format(base_dir, region)
src_dir = '/Users/pnorton/Projects/National_Hydrology_Model/regions/{0:s}_byHRU'.format(region)
# dst_dir = '/Volumes/LaCie/20150720a/regions/%s_byHRU' % region

control_file = '{0:s}/daymet.control'.format(parent_dir)

# Read the control file
control = prms.control(control_file)

# Read the input parameter file
orig_params = prms.parameters('{0:s}/{1:s}'.format(parent_dir, control.get_var('param_file')['values'][0]))

param_filename = os.path.basename(control.get_var('param_file')['values'][0])

# TODO: Right now only nhru is handled for checking in changed parameters; this should be expanded for nssr and ngw
# split_dims = set(['nhru', 'nssr', 'ngw'])

# print('%s/%s_*/%s' % (src_dir, region, param_filename))
proc_list = [el for el in glob.glob('{0:s}/{1:s}_*/{2:s}'.format(src_dir, region, param_filename))]

for dd in proc_list:
    # Read the parameter file for the current HRU and get it's parent hru index
    new_params = prms.parameters(dd)
    parent_hru = new_params.get_var('nhm_id')['values'][0]

    # 2016-04-04 PAN: Going forward we will use the nhm_id to get the index into the original parameters
    index_to_orig = np.where(orig_params.nhm_id['values'] == parent_hru)[0][0] + 1

    sys.stdout.write('\r\tHRU: {0:06d} (orig_index={1:d}) '.format(parent_hru, index_to_orig - 1))
    sys.stdout.flush()

    # Only check-in the the parameters that were changed
    # Currently this is read from merged_params.txt
    for ii in mod_params:
        curr_param = new_params.get_var(ii)
        orig_params.update_values_by_hru(ii, curr_param['values'], index_to_orig)
print('')

orig_params.write_param_file('crap_new_param')
