
from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems, iterkeys

import xml.etree.ElementTree as xmlET
import pandas as pd

# from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.Parameters import Parameter
from pyPRMS.constants import REGIONS, PARAMETERS_XML, DIMENSIONS_XML, NHM_DATATYPES


class NhmParamDb(object):
    def __init__(self, paramdb_dir):
        self.__paramdb_dir = paramdb_dir

        self.__params = {}    # holds dictionary of parameters included in NhmParamDb repo

        self._global_params()
        # TODO: how to properly handle the xml-based parameter and dimension information
        # global_dims_file = '{}/dimensions.xml'.format(self.__paramdb_dir)

        # dimension_info = get_global_dimensions(param_info, REGIONS, workdir)

    @property
    def available_parameters(self):
        return self.__params

    def get(self, name):
        """Get a parameter from the NhmParamDb"""

        if name in self.__params.keys():
            curr_type = NHM_DATATYPES[self.__params[name]['type']]
            param = Parameter(name=name, datatype=curr_type)
            tmp_data = []

            for rr in REGIONS:
                # Read parameter information
                cdir = '{}/{}/{}'.format(self.__paramdb_dir, name, rr)

                # Add/grow dimensions for current parameter
                param.dimensions.add_from_xml('{}/{}.xml'.format(cdir, name))

                # Read the data
                fhdl = open('{}/{}.csv'.format(cdir, name))
                rawdata = fhdl.read().splitlines()
                fhdl.close()
                it = iter(rawdata)
                next(it)    # Skip the header row

                # Read the parameter values
                for rec in it:
                    idx, val = rec.split(',')
                    tmp_data.append(val)

            param.data = tmp_data
            return param
        raise KeyError('Parameter, {}, does not exist in the NHM parameter database'.format(name))

    def get_DataFrame(self, name):
        """Returns a pandas DataFrame for a parameter. If the parameter dimensions includes
           either nhrus or nsegment then the respective national ids are included as the
           index in the return dataframe"""

        param = self.get(name)
        param_data = param.data

        if set(param.dimensions.keys()).intersection(set(['nhru', 'ngw', 'nssr'])):
            param_hrus = self.get('nhm_id').data

            # Create a DataFrame of the parameter
            df = pd.DataFrame(param_data, index=param_hrus)
            df.index.name = 'nhm_id'
        elif set(param.dimensions.keys()).intersection(set(['nsegment'])):
            param_segments = self.get('nhm_seg').data

            # Create a DataFrame of the parameter
            df = pd.DataFrame(param_data, index=param_segments)
            df.index.name = 'nhm_seg'
        else:
            df = pd.DataFrame(param_data)

        if len(param_data.shape) == 2:
            df.rename(columns=lambda xx: '{}_{}'.format(name, df.columns.get_loc(xx) + 1), inplace=True)
        else:
            # Assuming 1D array
            df.rename(columns={0: name}, inplace=True)

        return df

    # def get_global_dimensions(params, regions, workdir):
    #     # This builds a dictionary of total dimension sizes for the concatenated parameters
    #     dimension_info = {}
    #     is_populated = {}
    #
    #     # Loop through the xml files for each parameter and define the total size and dimensions
    #     for pp in iterkeys(params):
    #         for rr in regions:
    #             cdim_tree = xmlET.parse('{}/{}/{}/{}.xml'.format(workdir, pp, rr, pp))
    #             cdim_root = cdim_tree.getroot()
    #
    #             for cdim in cdim_root.findall('./dimensions/dimension'):
    #                 dim_name = cdim.get('name')
    #                 dim_size = int(cdim.get('size'))
    #
    #                 is_populated.setdefault(dim_name, False)
    #
    #                 if not is_populated[dim_name]:
    #                     if dim_name in ['nmonths', 'ndays', 'one']:
    #                         # Non-additive dimensions
    #                         dimension_info[dim_name] = dimension_info.get(dim_name, dim_size)
    #                     else:
    #                         # Other dimensions are additive
    #                         dimension_info[dim_name] = dimension_info.get(dim_name, 0) + dim_size
    #
    #         # Update is_populated to reflect dimension size(s) don't need to be re-computed
    #         for kk, vv in iteritems(is_populated):
    #             if not vv:
    #                 is_populated[kk] = True
    #     return dimension_info

    def _global_params(self):
        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types
        params_file = '{}/{}'.format(self.__paramdb_dir, PARAMETERS_XML)

        # Read in the parameters.xml file
        params_tree = xmlET.parse(params_file)
        params_root = params_tree.getroot()

        for param in params_root.findall('parameter'):
            self.__params[param.get('name')] = {}
            self.__params[param.get('name')]['type'] = param.get('type')
            self.__params[param.get('name')]['units'] = param.get('units')

