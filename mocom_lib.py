#!/usr/bin/env python2.7


# Description: support library for MOCOM calibration
#     Created: 2015-09-15
#      Author: Parker Norton (pnorton@usgs.gov)

from __future__ import (absolute_import, division,
                        print_function)
# , unicode_literals)
from future.utils import iteritems

import pandas as pd
import prms_cfg as prms_cfg
import os
import re
# import shutil

__author__ = 'pnorton'


class opt_log(object):
    # Class to handle reading the mocom optimization log file
    # Current it only handles the native format of the log file
    # TODO: Add support to read the process csv version of the log file

    def __init__(self, filename, configfile=None):
        self.__optlog_data = None
        self.__filename = ''
        self.configfile = configfile
        self.filename = filename

    @property
    def configfile(self):
        # Return the current configfile being used or None
        return self.__configfile

    @configfile.setter
    def configfile(self, configfile):
        # Set/change the name of the configfile
        self.__configfile = configfile
        if self.__filename != '':
            self.read_log()

    @property
    def data(self):
        # Return the optimization log dataframe
        return self.__optlog_data

    @property
    def filename(self):
        # Return the current optimization log filename
        return self.__filename

    @filename.setter
    def filename(self, filename):
        # Set/change the name of optimization log filename
        self.__filename = filename
        self.read_log()

    @property
    def lastgen(self):
        if self.__filename != '':
            return max(self.__optlog_data['gennum'])
        else:
            return None

    @property
    def objfcnNames(self):
        # Return a list of the objective function column names
        return [col for col in self.__optlog_data.columns if 'OF_' in col]

    def get_modelrunids(self, generation):
        # Returns a list of modelrunids in a given generation
        # Can specify generation = 'last' or 'final' to get the last generation
        # Can specify generation = 'seed' to get the initial generation used to start calibration
        if isinstance(generation, int):
            return self.__optlog_data['soln_num'].loc[self.__optlog_data['gennum'] == generation].tolist()
        elif isinstance(generation, str):
            if generation in ['last', 'final']:
                return self.__optlog_data['soln_num'].loc[self.__optlog_data['gennum'] == self.lastgen].tolist()
            elif generation in ['seed']:
                return self.__optlog_data['soln_num'].loc[self.__optlog_data['gennum'] == 0].tolist()
        return None

    def read_log(self):
        # Read the optimization log file
        infile = open(self.__filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        bad_chars = '():='
        rgx = re.compile('[%s]' % bad_chars)

        tmp_data = []
        for line in it:
            if line[0:34] == 'Determining starting parameters...':
                # This is the group of random starting sets
                next(it)
                gennum = 0

                tmp_hdr = next(it).split()
                tmp_hdr.insert(0, 'setnum')
                hdr_flag = True

                while True:
                    try:
                        x = next(it)
                    except StopIteration:
                        break

                    if x[0:1] == '' or x[0:25] == 'MOCOM:  WARNING:  Current':
                        break

                    # Strip out the junk characters ():=
                    x = rgx.sub('', x) + ' ' + str(gennum)
                    x = x.split()
                    if x[1] == 'Bad':
                        continue

                    if hdr_flag:
                        # Header info from starting population is incomplete, fill it it out
                        # with information inferred from the first line of data
                        try:
                            cfg = prms_cfg.cfg(self.configfile)
                            ofunc = []

                            # Get the friendly name for each objective function
                            for (kk, vv) in iteritems(cfg.get_value('of_link')):
                                ofunc.append(vv['of_desc'])

                            # Populate the test columns with friendly OF names
                            for pp in range(0, (len(x) - len(tmp_hdr) - 3)):
                                tmp_hdr.append('OF_%s' % ofunc[pp])
                                # tmp_hdr.append('test%d' % pp)
                        except:
                            # No configfile specified so write generic objective function header
                            for pp in range(0, (len(x) - len(tmp_hdr) - 3)):
                                tmp_hdr.append('OF_%d' % pp)

                        # Add the remaining headers columns
                        tmp_hdr.append('rank')
                        tmp_hdr.append('soln_num')
                        tmp_hdr.append('gennum')

                        hdr_flag = False

                    # Append the data to optlog_data list
                    tmp_data.append(x)

            if line[0:34] == 'Current generation for generation ':
                gennum = int(line.split(' ')[-1].rstrip(':'))+1
                next(it)    # skip one line
                next(it)    # skip one line
                next(it)    # skip one line

                while True:
                    x = next(it)
                    if x[0:1] == '':
                        break

                    # Strip out the junk characters ():=
                    x = rgx.sub('', x) + ' ' + str(gennum)

                    tmp_data.append(x.split())
            elif line[0:48] == 'Results for multi-objective global optimization:':
                gennum = int(next(it).split()[1])+1
                next(it)    # skip one line
                next(it)    # skip one line
                next(it)    # skip one line

                while True:
                    x = next(it)
                    if x[0:1] == '':
                        break

                    # Strip out the junk characters ():=
                    x = rgx.sub('', x) + ' ' + str(gennum)

                    tmp_data.append(x.split())

        # Create a dataframe from the imported optimization log data
        self.__optlog_data = pd.DataFrame(tmp_data, columns=tmp_hdr).apply(pd.to_numeric, errors='coerce')

    def remove_nonpareto_directories(self, modelrunid, keep_seed=True):
        # Create list of winner runids padded with zeros to match the directory listing
        # If keep_seed is true then add those to the winners list
        try:
            cfg = prms_cfg.cfg(self.configfile)

            modelrun_dir = '%s/%s/runs/%s' % (cfg.base_calib_dir, cfg.get_value('basin'), modelrunid)
        except:
            print("ERROR: runs/%s doesn't exist!" % modelrunid)
            return

        winners = []
        for rr in self.get_modelrunids('last'):
            winners.append('%05d' % rr)

        if keep_seed:
            for rr in self.get_modelrunids('seed'):
                winners.append('%05d' % rr)

        # Get the directory listing of runids from the calibration run
        stdir = os.getcwd()

        os.chdir(modelrun_dir)
        directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]

        # Do a little set magic to find the directories that are not in the winners list.
        # These are the directories we can safely delete
        remove_list = list(set(directories) - set(winners))

        print("Removing:", remove_list)
        # for dd in remove_list:
        #     shutil.rmtree(dd)

        # Return to starting directory
        os.chdir(stdir)

    def write_csv(self, filename):
        # Write the optimization log out to a csv file
        self.__optlog_data.to_csv(filename, index=False, header=True)