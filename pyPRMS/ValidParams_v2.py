
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import glob
import os
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.constants import PARNAME_DATATYPES


class ValidParams_v2(ParameterSet):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-01-06
    # Description: Object for the database of valid input parameters

    def __init__(self, filename):
        super(ValidParams_v2, self).__init__()

        # Mapping of certain parameters to their correct module
        self.__mod_map = {'hru_pansta': ['potet_pan'],
                          'tmin_adj': ['temp_1sta', 'temp_laps', 'ide_dist', 'xyz_dist'],
                          'tmax_adj': ['temp_1sta', 'temp_laps', 'ide_dist', 'xyz_dist'],
                          'hru_tsta': ['temp_1sta', 'temp_laps'],
                          'basin_tsta': ['temp_1sta', 'temp_laps', 'temp_dist2'],
                          'psta_elev': ['precip_laps', 'ide_dist', 'xyz_dist'],
                          'tsta_elev': ['temp_1sta', 'temp_laps', 'temp_dist2', 'ide_dist', 'xyz_dist'],
                          'elevlake_init': ['muskingum_lake'],
                          'gw_seep_coef': ['muskingum_lake'],
                          'lake_evap_adj': ['muskingum_lake'],
                          'lake_hru': ['muskingum_lake'],
                          'lake_hru_id': ['muskingum_lake'],
                          'lake_seep_elev': ['muskingum_lake'],
                          'lake_type': ['muskingum_lake']}

        self.__filename = filename
        # self.__paramdb = None

        # TODO: need more robust logic here; currently no way to handle failures
        self.__isloaded = False
        self._read()
        self.__isloaded = True

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, filename):
        self.__filename = filename

        self.__isloaded = False
        self._read()
        self.__isloaded = True

    # @property
    # def paramdb(self):
    #     return self.__paramdb

    def _read(self):
        """Read one or more par_name files to create a parameter set with no data"""
        filelist = []

        if os.path.isfile(self.__filename):
            # A single input par_name file was specified
            filelist = [self.__filename]
        elif os.path.isdir(self.__filename):
            # A path containing multiple par_name files was specified
            filelist = [el for el in glob.glob('%s/*par_name*' % self.__filename)]

        for ff in filelist:
            curr_dict = self.__read_parname_file(ff)

            for (pname, vv) in iteritems(curr_dict):
                try:
                    self.parameters.add(pname)

                    try:
                        self.parameters.get(pname).units = vv['units']
                    except KeyError:
                        print('KeyError: {} has no units'.format(pname))

                    self.parameters.get(pname).datatype = PARNAME_DATATYPES[vv['datatype']]
                    self.parameters.get(pname).description = vv['desc']
                    self.parameters.get(pname).help = vv['help']

                    self.parameters.get(pname).modules = vv['module']
                    self.parameters.get(pname).minimum = vv['min']
                    self.parameters.get(pname).maximum = vv['max']
                    self.parameters.get(pname).default = vv['default']

                    for dd in vv['dimensions']:
                        # Add dimensions to the parameter
                        self.parameters.get(pname).dimensions.add(dd)

                        # Add dimensions to the global dimensions object
                        # self.dimensions.add(dd)

                    # TODO: add other parameter attributes
                except ParameterError:
                    # Parameter exists add any new attribute information
                    pass

    # def __build_paramdb(self):
    #     """Build the input parameter db from a collection of par_name files in a directory"""
    #     filelist = []
    #
    #     if os.path.isfile(self.__filename):
    #         # A single input par_name file was specified
    #         filelist = [self.__filename]
    #     elif os.path.isdir(self.__filename):
    #         # A path containing multiple par_name files was specified
    #         filelist = [el for el in glob.glob('%s/*par_name*' % self.__filename)]
    #
    #       thefirst = True
    #
    #     for ff in filelist:
    #         if thefirst:
    #             self.__paramdb = self.__read_parname_file(ff)
    #             thefirst = False
    #         else:
    #             curr_dict = self.__read_parname_file(ff)
    #
    #             # Add new control parameters or update module field for existing parameters
    #             for (kk, vv) in iteritems(curr_dict):
    #                 if kk in self.__paramdb:
    #                     # Control parameter already exists, check if this is a new module
    #                     if isinstance(self.__paramdb[kk]['module'], list):
    #                         if vv['module'][0] not in self.__paramdb[kk]['module']:
    #                             self.__paramdb[kk]['module'] += vv['module']
    #                             # self.__paramdb[kk]['Module'].append(vv['Module'])
    #                     else:
    #                         if vv['module'] != self.__paramdb[kk]['module']:
    #                             # Convert Module entry to a list and add the new module name
    #                             tmp = self.__paramdb[kk]['module']
    #                             self.__paramdb[kk]['module'] = [tmp, vv['module']]
    #                 else:
    #                     # We have a new control parameter
    #                     self.__paramdb[kk] = vv

    def __read_parname_file(self, filename):
        """Given a .par_name file (generated by prms -print) returns a dictionary
           of valid parameters. Returns None if file cannot be opened"""

        validparams = {}

        # Create parameter default ranges file from from PRMS -print results
        try:
            infile = open(filename, 'r')
        except IOError as err:
            print("Unable to open file\n", err)
            return None
        else:
            rawdata = infile.read().splitlines()
            infile.close()

            it = iter(rawdata)

            # Process dimensions first
            for line in it:
                if line == '--------------- DIMENSIONS ---------------':
                    break

            for line in it:
                flds = line.split(':')

                if line == '--------------- PARAMETERS ---------------':
                    break

                if len(flds) < 2:
                    continue

                key = flds[0].strip().lower()
                val = flds[1].strip()

                if key == 'name':
                    cdim = val

                    # Add dimensions to the global dimensions object
                    self.dimensions.add(val)
                elif key == 'desc':
                    self.dimensions.get(cdim).description = val
                else:
                    pass

            # for line in it:
            #     if line == '--------------- PARAMETERS ---------------':
            #         break

            toss_param = False  # Trigger for removing unwanted parameters

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Each parameter entry in a *.par_name file follows this format:
            # --------------------------------------------------------------
            # Name: adjmix_rain
            # Module: precip_1sta
            # Descr: Adjustment factor for rain in a rain / snow mix
            # Help: Monthly(January to December) factor to adjust rain proportion in a mixed rain / snow event
            # Ndimen: 2
            # Dimensions: nhru - 55, nmonths - 12
            # Size: 660
            # Type: float
            # Units: decimal fraction
            # Width: 0
            # Max: 3.0
            # Min: 0.0
            # Default: 1.0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            for line in it:
                flds = line.split(':')

                if len(flds) < 2:
                    continue

                key = flds[0].strip().lower()
                val = flds[1].strip()

                # Only need 'Name' and 'Module' information
                if key == 'name':
                    if toss_param:
                        # Remove prior parameter if it was not wanted
                        del validparams[cparam]
                        toss_param = False

                    cparam = val  # Save parameter name for the remaining information
                    validparams[cparam] = {}
                elif key == 'module':
                    # TODO: module='setup' should be included for generating an official
                    # set of parameters but should not be included when generating
                    # the set of parameters for verifying a parameter file. Not
                    # sure how to handle this yet. Can uncomment the following
                    # 3 lines to strip 'setup' related parameters.
                    # if val == 'setup':
                    #     # Don't want to include parameters from the setup module
                    #     toss_param = True

                    # Override module(s) for select parameters
                    if cparam in self.__mod_map:
                        validparams[cparam][key] = self.__mod_map[cparam]
                    else:
                        validparams[cparam][key] = [val]
                elif key == 'ndimen':
                    # Number of dimensions is superfluous; don't store
                    pass
                elif key == 'descr':
                    # Use standard key for description
                    validparams[cparam]['desc'] = val
                elif key == 'dimensions':
                    # Get the dimension names; discard the sizes
                    dnames = [xx.split('-')[0].strip() for xx in val.split(',')]
                    validparams[cparam][key] = dnames
                elif key == 'size':
                    # Don't need the total parameter size
                    pass
                elif key == 'type':
                    cparam_type = val  # needed to convert max, min, and default values
                    validparams[cparam]['datatype'] = val
                elif key == 'units':
                    if cparam_type == 'string':
                        # Units for strings are 'none'; no reason to store
                        pass
                    else:
                        validparams[cparam][key] = val
                elif key == 'width':
                    # Width currently isn't populated
                    pass
                elif key in ['max', 'min', 'default']:
                    validparams[cparam][key] = val
                else:
                    validparams[cparam][key] = val
        return validparams

    # def get_param_subset(self, mods):
    #     """Return subset of paramdb based on selected modules"""
    #     subset = {}
    #
    #     param_by_module = self.module_params(mods)
    #
    #     for (cmod, params) in iteritems(param_by_module):
    #         for param in params:
    #             subset[param] = self.__paramdb[param]
    #
    #     return subset
    #
    # def module_params(self, mod):
    #     # mod is a dictionary containing a single entry with format:
    #     #         key = a valid module name for any of vals
    #     #         val = one of [et_module, precip_module, solrad_module, srunoff_module,
    #     #               strmflow_module, temp_module, transp_module]
    #
    #     params_by_module = {}
    #
    #     # Build params by modules
    #     for (cmodname, val_mod) in iteritems(mod):
    #         # Can have one or more set
    #         for c_mod in val_mod:
    #             for (kk, vv) in iteritems(self.__paramdb):
    #                 if cmodname in vv['Module']:
    #                     if kk in ['potet_cbh_adj'] and cmodname == 'climate_hru' and c_mod != 'et_module':
    #                         # Only include potet_cbh_adj if et_module == climate_hru
    #                         continue
    #
    #                     if cmodname not in params_by_module:
    #                         # Add new module entry
    #                         params_by_module[cmodname] = []
    #
    #                     if kk not in params_by_module[cmodname]:
    #                         # Add new parameter name
    #                         params_by_module[cmodname].append(kk)
    #     return params_by_module
