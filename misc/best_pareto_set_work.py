#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems, itervalues

import argparse
import os
import pandas as pd
import tarfile
import prms_cfg
from prms_calib_helpers import get_sim_obs_stat
import mocom_lib as mocom


def select_files(members, file_list):
    for tarinfo in members:
        if os.path.basename(tarinfo.name) in file_list:
            tarinfo.name = os.path.basename(tarinfo.name)
            yield tarinfo


parser = argparse.ArgumentParser(description='Find the best pareto sets from a calibration run')
parser.add_argument('-c', '--config', help='Primary basin configuration file', required=True)
parser.add_argument('-r', '--runid', help='Runid of the calibration to display', required=True)

args = parser.parse_args()

configfile = args.config
runid = args.runid

basinConfigFile = 'basin.cfg'

# runid = '2016-04-06_1609'
# configfile = '/media/scratch/PRMS/calib_runs/pipestem_1/basin.cfg'

# Read the calibration configuration file
cfg = prms_cfg.cfg(configfile)

print(os.getcwd())
print('runid:', runid)

# Get the list of basins for this calibration
basins = cfg.get_basin_list()
print('Total basins: {0:d}'.format(len(basins)))

base_dir = cfg.base_dir

merged_df = None
test_group = ['OF_AET', 'OF_SWE', 'OF_runoff', 'OF_comp']
# test_group = ['OF_AET', 'OF_SWE', 'OF_comp']
test_out = []

cdir = os.getcwd()
print('Starting directory: %s' % cdir)

for bb in basins:
    workdir = '{0:s}/{1:s}/{2:s}'.format(base_dir, runid, bb)
    basin_config_file = '{0:s}/{1:s}'.format(workdir, basinConfigFile)
    basin_cfg = prms_cfg.cfg(basin_config_file)

    # TODO: Check for .success file before including an HRU
    if not (os.path.isfile('{0:s}/.success'.format(workdir)) or os.path.isfile('{0:s}/.warning'.format(workdir))):
        continue

    hrunum = int(bb.split('_')[1]) + 1
    
    # Read in mocom file
    mocom_log = mocom.opt_log(basin_cfg.get_log_file(runid), basin_config_file)
    objfcns = mocom_log.objfcnNames
    lastgen_data = mocom_log.data[mocom_log.data['gennum'] == mocom_log.lastgen]

    # The minimum composite OF result is used to select the pareto set member
    # for each HRU that will be merge back into the parent region
    lastgen_data.loc[:, 'OF_comp'] = 0.45 * lastgen_data['OF_AET'] + 0.45 * lastgen_data['OF_SWE'] + \
                                     0.1 * lastgen_data['OF_runoff']
    # lastgen_data.loc[:, 'OF_comp'] = 0.5*lastgen_data['OF_AET'] + 0.5*lastgen_data['OF_SWE']
    
    print(hrunum,)
    for tt in test_group:
        # Get the set with the best NRMSE for the current test_group OF
        best = lastgen_data[lastgen_data[tt] == lastgen_data[tt].min()]['soln_num'].values

        if len(best) == 1:
            csoln = '{0:05d}'.format(best[0])
        else:
            # We probably got multiply matches to the minimum value so
            # use AET (for SWE), SWE (for AET), or SWE (for runoff)
            # as a tie breaker
            csoln = 0
            print('tie-breaker!', hrunum, tt)
            tmp1 = lastgen_data[lastgen_data[tt] == lastgen_data[tt].min()]

            if tt == 'OF_SWE':
                csoln = '{0:05d}'.format(tmp1[tmp1['OF_AET'] == tmp1['OF_AET'].min()]['soln_num'].values[0])
            elif tt == 'OF_AET':
                csoln = '{0:05d}'.format(tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]['soln_num'].values[0])
            elif tt == 'OF_runoff':
                csoln = '{0:05d}'.format(tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]['soln_num'].values[0])

        print('({0:d}) For {1:s} best solution is: {2:s}'.format(hrunum, tt, csoln))

        soln_workdir = '{0:s}/{1:s}'.format(workdir, csoln)
        try:
            os.chdir(soln_workdir)
        except:
            # Always want to end up back where we started
            print('Awww... crap!')
            os.chdir(cdir)

        # Get a list of the objective function observation files we will need
        obs_list = [vv['obs_file'] for vv in itervalues(cfg.get_value('objfcn'))]

        # Extract the observation files from the region archive into the current solution directory
        tar = tarfile.open('{}/{}.tar.gz'.format(cfg.get_value('template_dir'), bb))
        tar.extractall(members=select_files(tar, obs_list), path='.')
        tar.close()

        outputstats = []
        objfcn_link = cfg.get_value('of_link')

        # Change the objective functions to percent bias of the mean monthly interval
        for (kk, vv) in iteritems(cfg.get_value('objfcn')):
            cfg.change_objfcn_stat(kk, 'PBIAS')
            cfg.change_objfcn_interval(kk, 'mnmonth')

        tmp_data = [hrunum, csoln, tt]

        for vv in itervalues(objfcn_link):
            tmp_data.append(get_sim_obs_stat(cfg, vv, verbose=False))

        # Add results to the test_out list
        test_out.append(tmp_data)
    os.chdir(cdir)

# Create dataframe of results
ofnames = mocom_log.objfcnNames
ofnames.insert(0, 'best')
ofnames.insert(0, 'soln_num')
ofnames.insert(0, 'HRU')
df_results = pd.DataFrame(test_out, columns=ofnames)

# write to a csv file
df_results.to_csv('%s_best.csv' % runid)
df_results.head()

# In[ ]:
bb = df_results[df_results.best == 'OF_AET']
print('Best AET (min/max/median/mean): {0:0.1f}/{1:0.1f}/{2:0.1f}/{3:0.1f}'
      .format(bb.OF_AET.min(), bb.OF_AET.max(), bb.OF_AET.median(), bb.OF_AET.mean()))

bb = df_results[df_results.best == 'OF_SWE']
print('Best SWE (min/max/median/mean): {0:0.1f}/{1:0.1f}/{2:0.1f}/{3:0.1f}'
      .format(bb.OF_SWE.min(), bb.OF_SWE.max(), bb.OF_SWE.median(), bb.OF_SWE.mean()))

bb = df_results[df_results.best == 'OF_comp']
print('Best Composite: SWE (min/max/median/mean): {0:0.1f}/{1:0.1f}/{2:0.1f}/{3:0.1f}'
      .format(bb.OF_SWE.min(), bb.OF_SWE.max(), bb.OF_SWE.median(), bb.OF_SWE.mean()))

bb = df_results[df_results.best == 'OF_comp']
print('Best Composite: AET (min/max/median/mean): {0:0.1f}/{1:0.1f}/{2:0.1f}/{3:0.1f}'
      .format(bb.OF_AET.min(), bb.OF_AET.max(), bb.OF_AET.median(), bb.OF_AET.mean()))
