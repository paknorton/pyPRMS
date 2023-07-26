
# import numpy as np
import os
import pandas as pd     # type: ignore
from typing import cast, Optional
# from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType, Set

from ..prms_helpers import read_xml
from .Parameters import Parameters
# from ..constants import DATATYPE_TO_DTYPE, NHM_DATATYPES
from ..constants import NEW_PTYPE_TO_DTYPE, PARAMETERS_XML, DIMENSIONS_XML


class ParamDb(Parameters):
    def __init__(self, paramdb_dir: str,
                 metadata,
                 verbose: Optional[bool] = False):
                 # verify: Optional[bool] = True):
        """Initialize ParamDb object.

        This object handles the monolithic parameter database.

        :param paramdb_dir: Path to the ParamDb directory
        :param verbose: Output additional debug information
        :param verify: Whether to load the master parameters (default=True)
        """

        super(ParamDb, self).__init__(metadata=metadata, verbose=verbose)
        self.__paramdb_dir = paramdb_dir
        self.__verbose = verbose

        # Read the parameters from the parameter database
        self._read()

    def _read(self):
        """Read a parameter database.
        """

        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types
        global_params_file = f'{self.__paramdb_dir}/{PARAMETERS_XML}'
        global_dimens_file = f'{self.__paramdb_dir}/{DIMENSIONS_XML}'

        # Read in the parameters.xml and dimensions.xml file
        params_root = read_xml(global_params_file)
        dimens_root = read_xml(global_dimens_file)

        # Populate the global dimensions from the xml file
        for xml_dim in dimens_root.findall('dimension'):
            self.dimensions.add(name=cast(str, xml_dim.attrib.get('name')), size=cast(int, xml_dim.find('size').text))

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = cast(str, param.attrib.get('name'))
            curr_file = f'{self.__paramdb_dir}/{xml_param_name}.csv'

            if self.exists(xml_param_name):
                # Sometimes the global parameter xml file has duplicates of parameters
                print(f'WARNING: {xml_param_name} is duplicated in {PARAMETERS_XML}; skipping')
                continue

            if os.path.exists(curr_file):
                self.add(xml_param_name)

                cdtype = NEW_PTYPE_TO_DTYPE[self.get(xml_param_name).meta['datatype']]
                tmp_data = pd.read_csv(curr_file,
                                       skiprows=0,
                                       usecols=[1],
                                       dtype={1: cdtype}).squeeze('columns').to_numpy()

                self.get(xml_param_name).data = tmp_data
            else:
                print(f'WARNING: {xml_param_name}, ParamDb file does not exist; skipping')
