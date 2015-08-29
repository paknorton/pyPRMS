#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
import pandas as pd
import datetime
import calendar
import operator
import prms_cfg as prms_cfg
import prms_lib as prms
import prms_objfcn as objfcn

__version__ = 0.5


def dparse(*dstr):
    dint = [int(x) for x in dstr]

    if len(dint) == 2:
        # For months we want the last day of each month
        dint.append(calendar.monthrange(*dint)[1])
    if len(dint) == 1:
        # For annual we want the last day of the year
        dint.append(12)
        dint.append(calendar.monthrange(*dint)[1])

    return datetime.datetime(*dint)


def link_data(src, dst):
    """Symbolically link a file into a directory. This adds code to handle
       doing this over an NFS mount. The src parameter should be a path/filename
       and the dst parameter should be a directory"""

    dst_file = '%s/%s' % (dst, os.path.basename(src))

    if os.path.isfile(dst_file):
        # Remove any pre-existing file/symlink

        # These operations over NFS sometimes generate spurious ENOENT and
        # EEXIST errors. To get around this just rename the link/file first
        # and then delete the renamed version. This works because rename is
        # atomic whereas unlink and remove are not.
        tmpfile = '%s/tossme' % dst
        os.rename(dst_file, tmpfile)
        try:
            if os.path.islink(tmpfile):
                os.unlink(tmpfile)
            else:
                os.remove(tmpfile)
        except OSError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            if not os.path.exists(tmpfile):
                print "\tHmmm... file must be gone already."

    # Now create the symbolic link
    os.symlink(src, dst_file)


# Related parameters
# These parameters need to satisfy relationships before PRMS is allowed to run
related_params = {'soil_rechr_max': {'soil_moist_max': operator.le},
                  'tmax_allsnow': {'tmax_allrain': operator.lt}}


months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Command line arguments
parser = argparse.ArgumentParser(description='Post MOCOM processing')
parser.add_argument('mocomrun', help='MOCOM run id')
parser.add_argument('modelrun', help='PRMS model run id')
parser.add_argument('parameters', help='Parameters calculated by MOCOM',
                    nargs=argparse.REMAINDER, type=float)

args = parser.parse_args()

# This is set as a hardcoded default for now
configfile = 'basin.cfg'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read the config file
cfg = prms_cfg.cfg(configfile)

cdir = os.getcwd()  # Save current dir in case we need it later

basin_dir = '%s/%s' % (cfg.get_value('base_calib_dir'), cfg.get_value('basin'))
run_dir = '%s/%s/%s/%s' % (basin_dir, cfg.get_value('runs_sub'), args.mocomrun, args.modelrun)

# Make the run directory and change into it
try:
    os.makedirs(run_dir)
except OSError:
    # The args.modelrun directory already exists. Delete it and re-create it.
    shutil.rmtree(run_dir)
    os.makedirs(run_dir)
finally:
    os.chdir(run_dir)

# Copy the control and input parameter files into calibration directory
for dd in cfg.get_value('source_config'):
    cmd_opts = " %s ." % dd
    subprocess.call(cfg.get_value('cmd_cp') + cmd_opts, shell=True)

# Link in the data files
for dd in cfg.get_value('source_data'):
    link_data(dd, os.getcwd())

# Read the param_range_file
# Check that the number of parameters in the file matches the parameters
# passed in the command line arguments.
# The param_range_file is located in the root basin directory
pfile = open('%s/%s' % (cdir, cfg.get_value('param_range_file')), 'r')
params = []
for rr in pfile:
    params.append(rr.split(' ')[0])
pfile.close()

if len(params) != len(args.parameters):
    print "ERROR: mismatch between number of parameters (%d vs. %d)" % (len(params), len(args.parameters))
    exit(1)


# ************************************
# Check related parameters here
# ************************************
for ii,pp in enumerate(params):
    if pp in related_params:
        for kk, vv in related_params[pp].iteritems():
            if kk in params:
                # Test if the parameter meets the condition for the related parameter
                if not vv(args.parameters[ii], args.parameters[params.index(kk)]):
                    # Write the stats.txt file with -1 for each objective function and
                    # skip running PRMS, return stats.txt with -1.0 for each objective function
                    print '%s = %s failed relationship with %s = %s' % (pp, args.parameters[ii], kk, args.parameters[params.index(kk)])
                    tmpfile = open('tmpstats', 'w')
                    objfcn_link = cfg.get_value('of_link')

                    # must use version of MOCOM that has been modified to detect
                    # -9999.0 as a bad set.
                    for kk, vv in objfcn_link.iteritems():
                        tmpfile.write('-9999.0\n')

                    tmpfile.close()

                    # Move the stats file to its final place - MOCOM looks for this file
                    os.rename('tmpstats', 'stats.txt')

                    # Return to the starting directory
                    os.chdir(cdir)

                    exit(0)     # We'll skip the model run and return success to MOCOM


# Check the distr_method in the config file
# If equal to 'distr_val' then:
#   strip trailing number off of each parameter
#   find the parameter in the input parameter file
#   check if number of given parameters matches one of the dimension size for the input parameter
#       - if not ERROR
#   loop through ndimensions and copy new values to parameter
#
# If equal to 'distr_mean' use the code below

# For each parameter, redistribute the mean to the original parameters
# and update the input parameter file
pobj = prms.parameters(cfg.get_value('prms_input_file'))

# Check if all params are identical
# This would be the case if calibrating a parameter using individual values
# e.g. tmax_allsnow when calibrating each month
if params.count(params[0]) == len(params):
    # check if number of params entries equals a dimension for that input parameter
    # params are assumed to be in sequential order (e.g. 1..12)
    pobj.replace_values(params[0], args.parameters)
else:
    # We're redistributing a mean value
    for ii, pp in enumerate(params):
        pobj.distribute_mean_value(pp, args.parameters[ii])

# Write the new values back to the input parameter file
pobj.write_param_file(cfg.get_value('prms_input_file'))

# Convert the config file start/end date strings to datetime
st_date_model = prms.to_datetime(cfg.get_value('start_date_model'))
st_date_calib = prms.to_datetime(cfg.get_value('start_date'))
en_date = prms.to_datetime(cfg.get_value('end_date'))

# Run PRMS
cmd_opts = " -C%s -set param_file %s -set start_time %s -set end_time %s" % (cfg.get_value('prms_control_file'),
                                                                             cfg.get_value('prms_input_file'),
                                                                             prms.to_prms_datetime(st_date_model),
                                                                             prms.to_prms_datetime(en_date))

subprocess.call(cfg.get_value('cmd_prms') + cmd_opts, shell=True)


# ***************************************************************************
# ***************************************************************************
# Generate the statistics for the run

# Get the name of the observation file
cobj = prms.control(cfg.get_value('prms_control_file'))
statvar_file = cobj.get_var('stat_var_file')['values'][0]

tmpfile = open("tmpstats", 'w')
outputstats = []

# Load the simulation data
sim_data = prms.statvar(statvar_file).data
sim_data = sim_data[st_date_calib:en_date]

# print '='*40
# print 'Read statvar data'

# Load the statvar dataframe
# Range files from Lauren use -99.0 as missing, other files use -999.0
missing = [-999.0, -99.0]

# Equate objfcn values to columns and order expected in the data file
colnm_lookup = {'range': ['obs_lower', 'obs_upper'],
                'value': ['obs_val'],
                'daily': ['year', 'month', 'day'],
                'monthly': ['year', 'month'],
                'annual': ['year'],
                'mnmonth': ['month']}

objfcn_link = cfg.get_value('of_link')
of_dict = cfg.get_value('objfcn')

for kk, vv in objfcn_link.iteritems():
    of_result = 0

    for ii, of in enumerate(vv['of_names']):
        curr_of = of_dict[of]

        # Get the total number of columns for the dtype and obs_intv and build the names to use for the dataframe.
        thecols = []
        thecols.extend(colnm_lookup[curr_of['obs_intv']])
        thecols.extend(colnm_lookup[curr_of['obs_type']])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read in the observation values/ranges
        if curr_of['obs_intv'] == 'mnmonth':
            # The index won't be a datetime, instead it's a month value
            df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                              header=None, na_values=missing, names=thecols, index_col=0)
        else:
            # NOTE: When parsing year-month dates pandas defaults to the 21st of each month. I'm not sure yet
            #       if this will cause a problem.
            #       Annual dates are parsed as Jan-1 of the given year.
            # TODO: if 'obsfile' == statvar then read the observed values in from the statvar file
            df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                              header=None, na_values=missing, date_parser=dparse,
                              names=thecols, parse_dates={'thedate': colnm_lookup[curr_of['obs_intv']]}, index_col='thedate')

            if curr_of['obs_intv'] == 'monthly':
                df1 = df1.resample('M', how='mean')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Merge simulated with observed; resample simulated if necessary
        if curr_of['obs_intv'] == 'daily':
            df1_join_sim = df1.join(sim_data.loc[:,curr_of['sim_var']], how='left')
        else:
            if curr_of['obs_intv'] == 'monthly':
                if curr_of['sim_var'] in ['hru_actet']:
                    # This is for variables that should be summed instead of averaged
                    # FIXME: make this dynamic - maybe embed in basin.cfg?
                    tmp = sim_data.loc[:,curr_of['sim_var']].resample('M', how='sum')
                else:
                    tmp = sim_data.loc[:,curr_of['sim_var']].resample('M', how='mean')
            elif curr_of['obs_intv'] == 'mnmonth':
                monthly = sim_data.loc[:,curr_of['sim_var']].resample('M', how='mean')
                tmp = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
            elif curr_of['obs_intv'] == 'annual':
                tmp = sim_data.loc[:,curr_of['sim_var']].resample('A-SEP', how='mean')
            else:
                print "ERROR"
                tmp = None
                exit(1)
            df1_join_sim = df1.join(tmp, how='left')

        df1_join_sim.rename(columns = {curr_of['sim_var']: 'sim_var'}, inplace=True)

        # =================================================================
        # Read in the subdivide data, if specified
        if curr_of['sd_file'] is not None:
            # The subdivide file must be a daily timestep
            thecols = ['year', 'month', 'day', 'sdval']

            # Read the subdivide data
            df2 = pd.read_csv(curr_of['sd_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                              header=None, na_values=missing,
                              names=thecols, parse_dates={'thedate': ['year', 'month', 'day']}, index_col='thedate')

            # Merge the subdivide data with the observed data
            if curr_of['obs_intv'] != 'daily':
                # The observed data is not a daily timestep (subdivide data is daily) so raise an error.
                print 'ERROR: observed data must be daily timestep when using subdivide data'
                exit()

            # Merge statvar and observed data
            df_final = df1_join_sim.join(df2, how='left')

            # Subset to only include values which match 'sdval'
            df_final = df_final[df_final['sdval'] == curr_of['sd_val']]
        else:
            df_final = df1_join_sim

        # -----------------------------------------------------------------
        # Now resample to specified of_intv
        if curr_of['of_intv'] == 'monthly':
            # We only want to include complete months
            df_final = objfcn.get_complete_months(df_final)

            df_final = df_final.resample('M', how='mean')
        elif curr_of['of_intv'] == 'annual':
            # We only want to include complete water years
            df_final = objfcn.get_complete_wyears(df_final)

            # TODO: For now the annual interval is assumed to be water-year based
            df_final = df_final.resample('A-SEP', how='mean')
        elif curr_of['of_intv'] == 'mnmonth':
            # We only want to include complete months
            df_final = objfcn.get_complete_months(df_final)

            monthly = df_final.resample('M', how='mean')
            df_final = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
        elif curr_of['of_intv'] in months:
            # We are working with a single month over the time period
            df_final = df_final[df_final.index.month==(months.index(curr_of['of_intv'])+1)]

            # TODO: strip rows with NaN observations out of dataframe
        df_final = df_final.dropna(axis=0, how='any', thresh=None, inplace=False).copy()

        # ** objective function looks for sim_val for simulated and either obs_val or obs_lower, obs_upper
        of_result += vv['of_wgts'][ii] * objfcn.compute_objfcn(curr_of['of_stat'], df_final)
    # **** for of in vv['of_names']:

    print '%s: %0.6f' % (vv['of_desc'], of_result)
    tmpfile.write('%0.6f ' % of_result)
# **** for kk, vv in objfcn_link.iteritems():

tmpfile.write('\n')
tmpfile.close()

# Move the stats file to its final place - MOCOM looks for this file
os.rename('tmpstats', 'stats.txt')

# NOTE: Could extend this to generate all sorts of statistics that aren't
#        necessarily written to the mocom stats.txt file.
# ---------------------------------------------------------------------------

# Return to the starting directory
os.chdir(cdir)

