from __future__ import annotations

import os
import pandas as pd   # type: ignore

from collections.abc import Sequence

from ..constants import MetaDataType
from .InputVariable import InputVariable
from ..parameters.Parameters import Parameters
from ..base.console import get_console_instance

con = None

HEADER_SEP = '////'
STATION_START = '// Station'
UNITS_START = '// Unit:'
DATA_SEP = '####'
COMMENT = '//'
# NA_VALS_DEFAULT = ('-99.0', '-999.0')


class DataFile(object):
    """Class for working with PRMS ASCII input data files
    """

    def __init__(self, filename: str | os.PathLike,
                 metadata: MetaDataType,
                 parameters: Parameters | None = None,
                 missing: Sequence[str] = ('-99.9', '-999.0', '-9999.0'),
                 verbose: bool = False):
        """Create the DataFile object.

        :param filename: name of data file
        :param metadata: Metadata for the data file variables
        :param parameters: Parameters object
        :param missing: list of missing values
        :param verbose: output debugging information
        """

        global con
        con = get_console_instance()

        self.__missing = missing
        self.filename = filename
        self.__verbose = verbose
        self.metadata = metadata['data_file']
        self.parameters = parameters

        self.__header = ''   # data file header from first line of the file
        self.__station_meta_header: list[str] = []   # header lines before station metadata (if provided)
        self.__df_file_metadata: pd.DataFrame | None = None

        # Dictionary of input variables and InputVariable objects
        self.__input_vars: dict[str, InputVariable] = {}

        # Internal dictionary of input variables and associated file metadata
        self.__input_vars_intern: dict[str, dict[str, int | str | list[str]]] = {}

        self.__data_raw: pd.DataFrame | None = None
        self.__data_combined: pd.DataFrame | None = None

        self.load_file(self.filename)
        self._resolve_units()

    @classmethod
    def from_file(cls, filename: str | os.PathLike,
                  metadata: MetaDataType,
                  parameters: Parameters | None = None,
                  missing: Sequence[str] = ('-99.9', '-999.0', '-9999.0'),
                  verbose: bool = False) -> DataFile:
        """Create a DataFile by reading from an ASCII data file.

        This is equivalent to calling ``DataFile(...)`` directly but makes the
        I/O step explicit.

        :param filename: name of data file
        :param metadata: Metadata for the data file variables
        :param parameters: Parameters object
        :param missing: list of missing values
        :param verbose: output debugging information
        :returns: DataFile instance
        """

        return cls(filename, metadata, parameters=parameters, missing=missing, verbose=verbose)

    def _resolve_units(self):
        """Resolve units for input variables using parameters or file metadata."""

        global con

        if self.parameters is None:
            # Resolve the units for each variable from the file units if possible
            for cvar in self.__input_vars.values():
                if '_units' in cvar.metadata['units']:
                    con.print(f'[dark_orange]WARNING[/]: {cvar.name} has units={cvar.metadata["units"]} '
                              f'but parameter was not supplied.')
        else:
            # Resolve the units of each variable from the parameter file
            self.resolve_units()

    def __repr__(self) -> str:
        """Concise string representation for debugging.

        :returns: String representation of the DataFile object
        """

        n_vars = len(self.__input_vars_intern)
        if self.__data_raw is not None:
            start = self.__data_raw.index.min()
            end = self.__data_raw.index.max()
            return f"DataFile(filename={self.filename!r}, variables={n_vars}, period={start} to {end})"
        return f"DataFile(filename={self.filename!r}, variables={n_vars})"

    @property
    def data(self) -> pd.DataFrame:
        """Pandas dataframe of the data file for each input variable.

        The result is cached after the first access. Call :meth:`invalidate_cache`
        if the underlying variable data has been modified.

        :returns: Pandas dataframe of the data file
        """
        if self.__data_combined is None:
            frames = []
            for cvar in self.input_variables.keys():
                tmp_df = self.get(cvar).data.copy()
                tmp_df.rename(columns=self.get(cvar).full_column_names, inplace=True)
                frames.append(tmp_df)
            self.__data_combined = pd.concat(frames, axis=1)
        return self.__data_combined

    def invalidate_cache(self) -> None:
        """Invalidate the cached combined DataFrame.

        Call this after modifying individual input variable data so that the
        next access to :attr:`data` rebuilds the combined frame.
        """
        self.__data_combined = None

    @property
    def file_metadata(self) -> pd.DataFrame:
        return self.__df_file_metadata

    @property
    def header(self):
        return self.__header

    @property
    def input_variables(self) -> dict[str, dict[str, int | str | list[str]]]:
        """Get the input variables in the data file.

        :returns: Dictionary of input variables that are available in the data file
        """

        return self.__input_vars_intern

    @property
    def station_meta_header(self):
        return self.__station_meta_header

    def resolve_units(self):
        """Adjust units metadata for input variables that have an initial units value of
        precip_units, runoff_units, or temp_units.

        :returns: None
        """

        selected_units = self.parameters.user_defined_units
        for cvar in self.__input_vars.values():
            if cvar.metadata['units'] in selected_units:
                cvar.metadata['units'] = selected_units[cvar.metadata['units']]

    def data_by_variable(self, variable: str) -> pd.DataFrame:
        """Get the data for a specific input variable

        :param variable: name of input variable
        :returns: Pandas dataframe of the data for the input variable
        """

        import warnings

        msg = "DataFile.data_by_variable() method to be deprecated"
        warnings.warn(msg, DeprecationWarning)
        data = self.__input_vars[variable].data.copy()

        # The names are a headache any other way, hard to make them truly
        # backwards compatible using the old code.
        data.columns = variable + "_" + data.columns
        assert type(data) is pd.DataFrame

        return data

    def get(self, name: str) -> InputVariable:
        """Get the metadata for a specific input variable.

        :param name: name of input variable
        :returns: InputVariable object
        """

        return self.__input_vars[name]

    def load_file(self, filename: str | os.PathLike):
        """Read the PRMS ASCII streamflow data file.

        :param filename: name of data file
        """

        header_info = []

        with open(filename, 'r') as fhdl:
            # First line is a descriptive header
            self.__header = fhdl.readline().rstrip()

            # Get the input variable names and sizes
            while line := fhdl.readline():
                line = line.rstrip()

                if len(line) == 0:
                    continue
                if line.startswith(COMMENT):
                    header_info.append(line)
                    continue
                if line.startswith(DATA_SEP):
                    break

                # Get the input variable name and total size for the variable
                nm: str
                sz: str | int

                nm, sz = tuple(line.split())
                sz = int(sz)

                if sz > 0:
                    if nm in self.__input_vars_intern:
                        raise KeyError(f'{nm} declared multiple times in the data file')
                    self.__input_vars_intern[nm] = dict(size=sz)

            # =============================
            # Process metadata
            self._add_file_metadata(header_info)

            # =============================
            # Read the input variables data
            # The first 6 columns are [year month day hour minute seconds]
            time_col_names = ['year', 'month', 'day', 'hour', 'minute', 'second']
            data_col_names = self._data_column_names()
            col_names = time_col_names.copy()
            col_names.extend(data_col_names)

            # Use pandas to read the data in from the remainder of the file
            self.__data_raw = pd.read_csv(fhdl, sep=r'\s+', header=None, na_values=self.__missing,
                                          names=col_names, engine='c', skipinitialspace=True)
            self.__data_raw['time'] = pd.to_datetime(self.__data_raw[time_col_names], yearfirst=True)
            self.__data_raw.drop(columns=time_col_names, inplace=True)
            self.__data_raw.set_index('time', inplace=True)

            # Add data to each input variable
            self._add_variable_data()

    def write_ascii(self, filename: str) -> None:
        """Write dataframe to ASCII formatted file.

        This routine always writes out missing data as -999
        """

        df = self.data.copy()

        out_order = [kk for kk in df.columns]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['second'] = df.index.second

        with open(filename, 'w') as outhdl:
            outhdl.write(f'{self.__header}\n')
            outhdl.write(f'{HEADER_SEP*15}\n')

            for xx in self.__station_meta_header:
                outhdl.write(f'{xx}\n')

            for kk in self.__input_vars.keys():
                for mstr in self.get(kk).file_metadata_str:
                    outhdl.write(f'{mstr}\n')

            outhdl.write(f'{HEADER_SEP*15}\n')

            for kk in self.__input_vars.keys():
                outhdl.write(f'{kk} {self.get(kk).num_stations}\n')

            outhdl.write(f'{DATA_SEP*15}\n')

            df.to_csv(outhdl, sep=' ', columns=out_order, index=False, header=False, na_rep='-999', encoding=None)

    def _add_file_metadata(self, header_info: list[str]):
        """Add file metadata from data file.

        :param header_info: list of header lines from the data file
        """

        if self.__verbose:
            con.print(f'{header_info=}')

        it = iter(header_info)
        line = next(it)

        try:
            while not line.startswith(STATION_START):
                line = next(it)

            # Process the station information
            self.__station_meta_header.append(line.strip())

            line = next(it)
            if line.lower().startswith('// id'):
                # Process the station information
                self.__station_meta_header.append(line.strip())
                meta_vars = line.replace(COMMENT, '').strip().split()

                self.__df_file_metadata = pd.DataFrame(columns=meta_vars)

                line = next(it)

                # Loop through the variables and read the station information for each one
                for kk, vv in self.__input_vars_intern.items():
                    if line.startswith(HEADER_SEP):
                        continue

                    for sz in range(vv['size']):
                        fields = line.replace(COMMENT, '').strip().lower().split()
                        stn_id = fields[0]

                        dup = 0
                        while stn_id in self.__input_vars_intern[kk].setdefault('stations', []):
                            con.print(f'[orange3]WARNING[/] {kk} station {stn_id} already declared in data file; adjusting variable name')
                            dup += 1
                            stn_id = f'{stn_id}dup{dup}'
                        fields[0] = stn_id

                        self.__input_vars_intern[kk].setdefault('stations', []).append(stn_id)
                        self.__input_vars_intern[kk].setdefault('file_metadata', []).append(line.strip())

                        try:
                            # Add metadata to file_metadata dataframe
                            self.__df_file_metadata.loc[len(self.__df_file_metadata)] = fields
                        except ValueError:
                            con.print(f'[red]ERROR[/]: file metadata for {kk} station {stn_id} has incorrect number of fields')
                            raise

                        line = next(it)

                # Check that the number of stations read for each variable matches the variable size
                for vv in self.__input_vars_intern.values():
                    if vv['size'] != len(vv['stations']):
                        con.print(f'[red]ERROR[/] Number of expected stations, {vv["size"]}, does not match number of stations read {vv["stations"]}')
        except StopIteration:
            con.print('[orange3]WARNING[/]: No usable metadata found')

            # We don't have useful metadata so we assign indexed station IDs
            self.__station_meta_header.append('// Station metadata:')
            self.__station_meta_header.append('// id type')

            self.__df_file_metadata = pd.DataFrame(columns=['id', 'type'])

            for kk, vv in self.__input_vars_intern.items():
                if 'stations' not in self.__input_vars_intern[kk]:
                    for sz in range(vv['size']):
                        self.__input_vars_intern[kk].setdefault('stations', []).append(f'{sz+1}')
                        self.__input_vars_intern[kk].setdefault('file_metadata', []).append(f'{COMMENT} {sz+1}')
                        self.__df_file_metadata.loc[len(self.__df_file_metadata)] = [f'{sz+1}', kk]

        try:
            while not line.startswith(UNITS_START):
                line = next(it)

            # Process the units
            while not line.startswith(HEADER_SEP):
                for elem in (line.replace(UNITS_START, '').replace(COMMENT, '').replace(' ', '').split(',')):
                    try:
                        cvar, cunits = elem.split('=')
                        try:
                            self.__input_vars_intern[cvar]['file_units'] = cunits
                        except KeyError:
                            con.print(f'[orange3]WARNING[/]: Units variable, {cvar}, is not a valid input variable name in this data file')
                    except ValueError:
                        con.print(f'[orange3]WARNING[/]: Malformed units information in data file')
                line = next(it)
        except StopIteration:
            con.print('[orange3]WARNING[/]: No unit information in data file metadata')

    def _add_variable_data(self):
        """Add data to each input variable.
        """

        # Create a data key for each input variable that maps to their respective parts of the dataframe
        st_idx = 0
        for cvar, cmeta in self.__input_vars_intern.items():
            self.__input_vars[cvar] = InputVariable(name=cvar,
                                                    data=self.__data_raw.iloc[:, st_idx:(st_idx + cmeta['size'])],
                                                    metadata=self.metadata,
                                                    station_metadata=self.__df_file_metadata.iloc[st_idx:(st_idx + cmeta['size'])],
                                                    file_units=cmeta.get('file_units', None))
            # self.__input_vars_intern[cvar]['data'] = self.__data_raw.iloc[:, st_idx:(st_idx + cmeta['size'])]
            st_idx += cmeta['size']

    def _data_column_names(self) -> list[str]:
        """Create column names for the dataframe.

        :returns: list of column names
        """

        var_col_names = []

        for cvar, meta in self.__input_vars_intern.items():
            if 'stations' in meta:
                for cstn in meta['stations']:   # type: ignore
                    var_col_names.append(f'{cvar}_{cstn}')
            else:
                # No usable metadata in the data file
                for idx in range(1, meta['size']+1):   # type: ignore
                    var_col_names.append(f'{cvar}_{idx}')

        return var_col_names
