#!/usr/bin/env python2.7


#      Author: Parker Norton (pnorton@usgs.gov)
#        Date: 2015-05-27
# Description: PRMS calibration configuration class which uses YAML on the backend


# import collections
import yaml
# from addict import Dict

class cfg(object):
    def __init__(self, filename):
        self.__cfgdict = None
        self.load(filename)


    @property
    def list(self):
        for kk,vv in self.__cfgdict.iteritems():
            print '%s:' % kk,

            if isinstance(vv, list):
                for ll in vv:
                    print ll,
                print
            else:
                print vv

    def add_var(self, var, value):
        # Add a new variable to the config dictionary
        # If the variable already exists, it will be overwritten
        if isinstance(value, list):
            self.__cfgdict[var] = value
        else:
            self.__cfgdict[var] = value


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
        yaml.dump(self.__cfgdict, open(filename, 'w'))

