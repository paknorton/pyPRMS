#!/usr/bin/env python2.7


#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2015-05-27
# Description: PRMS calibration configuration class which uses YAML on the backend

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

# import collections
import os
import re
import yaml
import pandas as pd
# from addict import Dict

# define the regex pattern that the parser will use to 'implicitely' tag your node
# {TMPDIR}
env_pattern = re.compile(r'^\<\'(.*)\'\>(.*)$')


def env_constructor(loader, node):
    value = loader.construct_scalar(node)

    try:
        env_var, remaining_path = env_pattern.match(value).groups()
        return os.environ[env_var] + remaining_path
    except KeyError:
        print('WARNING: Environment variable {0:s} does not exist; removing from value'.format(env_var))
        return remaining_path


class cfg(object):
    def __init__(self, filename, expand_vars=True):
        # Set expand_vars = False to not expand environment variables from config file
        self.__cfgdict = None

        if expand_vars:
            # See http://stackoverflow.com/questions/26712003/pyyaml-parsing-of-the-environment-variable-in-the-yaml-configuration-file
            # for more information

            # define a custom tag and associate with the regex pattern
            yaml.add_implicit_resolver('!env', env_pattern)

            # Register the constructor with yaml to handle environment variables
            yaml.add_constructor('!env', env_constructor)

        self.load(filename)

    @property
    def base_dir(self):
        return self.get_value('base_dir')

    @property
    def base_calib_dir(self):
        return self.get_value('base_calib_dir')

    @property
    def basins_file(self):
        return self.get_value('basins_file')

    @property
    def param_range_file(self):
        # Return full path to param_range_file
        try:
            return '{0:s}/{1:s}/{2:s}'.format(self.get_value('base_calib_dir'), self.get_value('basin'),
                                              self.get_value('param_range_file'))
        except KeyError:
            return '{0:s}/{1:s}'.format(self.get_value('base_calib_dir'), self.get_value('param_range_file'))

    def list_config_items(self):
        for (kk, vv) in iteritems(self.__cfgdict):
            print('{0:s}:'.format(kk)),

            if isinstance(vv, list):
                for ll in vv:
                    print(ll),
                print()
            else:
                print(vv)

    def add_var(self, var, value):
        # Add a new variable to the config dictionary
        # If the variable already exists, it will be overwritten
        if isinstance(value, list):
            self.__cfgdict[var] = value
        else:
            self.__cfgdict[var] = value

    def change_objfcn_stat(self, of_name, newval):
        if of_name in self.__cfgdict['objfcn']:
            self.__cfgdict['objfcn'][of_name]['of_stat'] = newval
        else:
            print('ERROR: {0:s} does not exist'.format(of_name))

    def change_objfcn_interval(self, of_name, newinterval):
        if of_name in self.__cfgdict['objfcn']:
            self.__cfgdict['objfcn'][of_name]['of_intv'] = newinterval
        else:
            print('ERROR: {0:s} does not exist'.format(of_name))

    def get_basin_list(self):
        # Returns a list of basin ids
        try:
            basinfile = open(self.get_value('basins_file'), 'r')
            basins = basinfile.read().splitlines()
            basinfile.close()
            return basins
        except:
            print("ERROR: unable to open/read {0:s}".format(self.get_value('basins_file')))
            return None

    def get_log_file(self, runid=None):
        if runid is None:
            try:
                return '{0:s}/{1:s}/{2:s}'.format(self.get_value('base_dir'), self.get_value('basin'),
                                                  self.get_value('log_file'))
            except:
                # TODO: assert an error
                return None
        else:
            try:
                return '{0:s}/{1:s}/{2:s}/{3:s}'.format(self.get_value('base_dir'), runid, self.get_value('basin'),
                                                        self.get_value('log_file'))
            except:
                # TODO: assert an error
                return None

    def get_param_limits(self):
        # Returns a dataframe of the contents of the param_range_file
        try:
            return pd.read_csv(self.param_range_file, header=None, names=['parameter', 'maxval', 'minval'],
                               sep=r'\s*', engine='python')
        except:
            print("ERROR: unable to read {0:s}".format(self.param_range_file))
            return None

    def get_value(self, varname):
        """Return the value for a given config variable"""
        return self.__cfgdict[varname]

    def load(self, filename):
        tmp = yaml.load(open(filename, 'r'))
        self.__cfgdict = tmp

    def update_value(self, variable, newval):
        """Update an existing configuration variable with a new value"""
        if variable in self.__cfgdict:
            self.__cfgdict[variable] = newval
        else:
            raise KeyError("Configuration variable does not exist")

    def write_config(self, filename):
        """Write current configuration information to a file"""

        # ***************************************************************************
        # Write basin configuration file for run
        yaml.safe_dump(self.__cfgdict, open(filename, 'w'), default_flow_style=False)
