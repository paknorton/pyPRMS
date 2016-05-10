#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# from future.utils import itervalues

# import os
import argparse
import subprocess

import prms_cfg
import prms_lib as prms
from prms_calib_helpers import read_default_params, read_sens_params, adjust_param_ranges

parser = argparse.ArgumentParser(description='Pre-processor for MOCOM optimization')
parser.add_argument('-C', '--config', help='Name of configuration file', required=True)
# parser.add_argument('--nocbh', help='Skip CBH file splitting', action='store_true')

args = parser.parse_args()

# ********next line for working with individual values*********
adj_ind_vals = False

# multiplier to influence number of sets
nsets_mult = 3
basin_cfg_file = args.config

cfg = prms_cfg.cfg(basin_cfg_file, expand_vars=True)

base_calib_dir = cfg.get_value('base_calib_dir')
basin = cfg.get_value('basin')
runid = cfg.get_value('runid')

# Location of the model data for this basin
model_src = '%s/%s' % (base_calib_dir, basin)
calib_job_dir = '%s/%s/%s' % (base_calib_dir, runid, basin)

paramfile = '%s/%s' % (model_src, cfg.get_value('prms_input_file'))
param_range_file = '%s/%s' % (calib_job_dir, cfg.get_value('param_range_file'))

# ==================================================================
# Verify the date ranges for the basin from the streamflow.data file
ctl = prms.control(cfg.get_value('prms_control_file'))

streamflow_file = '%s/%s' % (model_src, ctl.get_var('data_file')['values'][0])
first_date, last_date = prms.streamflow(streamflow_file).date_range

if first_date > prms.to_datetime(cfg.get_value('start_date_model')):
    print("\t* Adjusting start_date_model and start_date to reflect available streamflow data")
    cfg.update_value('start_date_model', first_date.strftime('%Y-%m-%d'))
    yr = int(first_date.strftime('%Y'))
    mo = int(first_date.strftime('%m'))
    dy = int(first_date.strftime('%d'))

    # Set calibration start date to 2 years after model start date
    cfg.update_value('start_date', '%d-%d-%d' % (yr+2, mo, dy))

if last_date < prms.to_datetime(cfg.get_value('end_date')):
    print("\t* Adjusting end_date to reflect available streamflow data")
    cfg.update_value('end_date', last_date.strftime('%Y-%m-%d'))

#     print "\tstart_date_model:", cfg.get_value('start_date_model')
#     print "\t      start_date:", cfg.get_value('start_date')
#     print "\t        end_date:", cfg.get_value('end_date')
# ------------------------------------------------------------------

# ==================================================================
# Create the param_range_file from the sensitive parameters file and the default ranges

# Some parameters should always be excluded even if they're sensitive
exclude_params = ['dday_slope', 'dday_intcp', 'jh_coef_hru', 'jh_coef']

# Some parameter should always be included
include_params = []

# Read the default parameter ranges from file
default_rng_filename = '%s/%s' % (calib_job_dir, cfg.get_value('default_param_list_file'))
def_ranges = read_default_params(default_rng_filename)

# Read the sensitive parameters in
sens_params = read_sens_params('%s/hruSens.csv' % calib_job_dir, include_params, exclude_params)

# ==================================================================
# Adjust the min/max values for each parameter based on the
# initial values from the input parameter file
adjust_param_ranges(paramfile, sens_params, def_ranges, param_range_file, make_dups=adj_ind_vals)

# Update the nparam and nsets values for the number of parameters we are calibrating
if adj_ind_vals:
    # Adjusting individual values for a parameter; Assumes a single param name
    cfg.update_value('nparam', sens_params.values()[0])
else:
    cfg.update_value('nparam', len(sens_params))

# Too many or too few of nsets will prevent calibration from converging
cfg.update_value('nsets', int(cfg.get_value('nparam') * nsets_mult))

# Set the number of tests according to the number of objective functions
cfg.update_value('ntests', len(cfg.get_value('of_link')))

# ***************************************************************************
# Write basin configuration file for run
cfg.write_config(basin_cfg_file)
# ---------------------------------------------------------------------------

# Run MOCOM with the following arguments:
#     nstart, nsets, nparam, ntests, calib_run, run_id, log_file, param_range_file, test_func_margin_file
mocom = cfg.get_value('cmd_mocom')
cmd_opts = ' %d %d %d %d %s %s %s %s %s' % (cfg.get_value('nstart'), cfg.get_value('nsets'), cfg.get_value('nparam'),
                                            cfg.get_value('ntests'), cfg.get_value('calib_run'), runid,
                                            cfg.get_value('log_file'), cfg.get_value('param_range_file'),
                                            cfg.get_value('test_func_margin_file'))
print("\tRunning MOCOM...")
print(cfg.get_value('cmd_mocom') + cmd_opts)
subprocess.call(mocom + cmd_opts, shell=True)

