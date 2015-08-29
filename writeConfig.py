#!/usr/bin/env python

import ConfigParser

config = ConfigParser.SafeConfigParser()

#START = (2000,01,01)
#END   = (2000,12,31)

config.add_section('General')
config.set('General', 'start_date','1985-10-1')
config.set('General', 'end_date','1990-9-30')

config.add_section('Calibration')
config.set('Calibration', 'nstart', '100')
config.set('Calibration', 'nsets', '25')
config.set('Calibration', 'nparam', '2')
config.set('Calibration', 'ntests', '3')

#config.add_section('Network')
#config.set('Network', 'username', 'someuser')
#config.set('Network', 'remote_host', 'ib-net')

config.set('DEFAULT', 'base_dir', '/Users/pnorton/Projects/National_Hydrology_Model/PRMS_testing')
config.set('DEFAULT', 'base_calib_dir', '%(base_dir)s/mocom_t1')

config.add_section('Paths')
config.set('Paths', 'template_dir', '%(base_dir)s/PRMS_master')
config.set('Paths', 'prms_control_sub', 'control')
config.set('Paths', 'prms_input_sub', 'input')
config.set('Paths', 'prms_output_sub', 'output')

config.add_section('Files')
config.set('Files', 'basins_file', '%(base_calib_dir)s/basin_list.txt')
config.set('Files', 'log_file', 'optim_log.log')
config.set('Files', 'param_range_file', 'param_limits.txt')
config.set('Files', 'test_func_margin_file', 'test_func_margins.txt')

config.add_section('Commands')
config.set('Commands', 'cp', '/bin/cp')
config.set('Commands', 'id', '/usr/bin/id')
config.set('Commands', 'ln', '/bin/ln')
config.set('Commands', 'mkdir', '/bin/mkdir')
config.set('Commands', 'mv', '/bin/mv')
config.set('Commands', 'ping', '/bin/ping')
config.set('Commands', 'rm', '/bin/rm')
config.set('Commands', 'scp', '/usr/bin/scp')
config.set('Commands', 'ssh', '/usr/bin/ssh')
config.set('Commands', 'prms', '%(base_calib_dir)s/bin/prms')
config.set('Commands', 'mocom', '%(base_calib_dir)s/bin/mocom')
config.set('Commands', 'calib_run', '%(base_calib_dir)s/optimize_wb_MASTER.sh')
config.set('Commands', 'stats_script', 'sflow_opti_stats_daily.pl')
config.set('Commands', 'plot_script', 'plot_day.ALL.csh')




# Writing our configuration file to 'example.cfg'
with open('example.cfg', 'w') as configfile:
    config.write(configfile)
