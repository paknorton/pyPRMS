from __future__ import annotations

import os
import pandas as pd     # type: ignore
from typing import cast

from ..constants import PRMS_VERSION
from ..Exceptions_custom import ParameterNotValidError
from ..prms_helpers import read_xml
from ..metadata import MetaData
from .Parameters import Parameters
from ..constants import NEW_PTYPE_TO_DTYPE, PARAMETERS_XML, DIMENSIONS_XML
from ..base.console import get_console_instance

con = None


class ParamDb(Parameters):
    def __init__(self, paramdb_dir: str,
                 metadata,
                 verbose: bool = False):
        """Initialize ParamDb object.

        This object handles the monolithic parameter database.

        :param paramdb_dir: Path to the ParamDb directory
        :param verbose: Output additional debug information
        """

        super().__init__(metadata=metadata, verbose=verbose)

        global con
        con = get_console_instance()

        self.__paramdb_dir = paramdb_dir
        self.__verbose = verbose

        # Read the parameters from the parameter database
        self._read()

    def _read(self):
        """Read a parameter database.
        """

        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types

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

        # Create a MetaData object to use its parameter parsing function
        mobj = MetaData()
        pvt_meta = mobj._parameters_to_dict(xml_root=params_root,
                                             meta_type='parameters',
                                             req_version=PRMS_VERSION)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = cast(str, param.attrib.get('name'))
            curr_file = f'{self.__paramdb_dir}/{xml_param_name}.csv'

            if self.exists(xml_param_name):
                # Sometimes the global parameter xml file has duplicates of parameters
                con.print(f'[orange3]WARNING[/]: {xml_param_name} is duplicated in {PARAMETERS_XML}; skipping')
                continue

            if os.path.exists(curr_file):
                try:
                    self.add(xml_param_name)
                except ParameterNotValidError:
                    con.print(f'[orange3]WARNING[/]: {xml_param_name} added custom metadata')
                    self.add_metadata(xml_param_name, pvt_meta[xml_param_name])
                    self.add(xml_param_name)

                cdtype = NEW_PTYPE_TO_DTYPE[self.get(xml_param_name).meta['datatype']]
                tmp_data = pd.read_csv(curr_file,
                                       skiprows=0,
                                       usecols=[1],
                                       dtype={1: cdtype}).squeeze('columns').to_numpy()

                self.get(xml_param_name).data = tmp_data
            else:
                con.print(f'[orange3]WARNING[/]: {xml_param_name}, ParamDb file does not exist; skipping')

        self.adjust_bounded_parameters()
