import os
import pandas as pd   # type: ignore

from typing import Dict, List, Optional, Sequence, Union

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

    def __init__(self, filename: Union[str, os.PathLike],
                 metadata: MetaDataType,
                 parameters: Parameters = None,
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
        self.__station_meta_header = []   # header lines before station metadata (if provided)
        self.__df_file_metadata: Optional[pd.DataFrame] = None

        # Dictionary of input variables and InputVariable objects
        self.__input_vars: Dict[str, InputVariable] = {}

        # Internal dictionary of input variables and associated file metadata
        self.__input_vars_intern: Dict[str, Dict[str, Union[int, str, List[str]]]] = {}

        self.__data_raw: Optional[pd.DataFrame] = None

        self.load_file(self.filename)

        if self.parameters is None:
            # Resolve the units for each variable from the file units if possible
            for cvar in self.__input_vars.values():
                if '_units' in cvar.metadata['units']:
                    con.print(f'[dark_orange]WARNING[/]: {cvar.name} has units={cvar.metadata["units"]} '
                              f'but parameter was not supplied.')
        else:
            # Resolve the units of each variable from the parameter file
            self.resolve_units()

    @property
    def data(self) -> pd.DataFrame:
        """Pandas dataframe of the data file for each input variable

        :returns: Pandas dataframe of the data file
        """

        return self.__data_raw

    @property
    def file_metadata(self) -> pd.DataFrame:
        return self.__df_file_metadata

    @property
    def header(self):
        return self.__header

    @property
    def input_variables(self) -> Dict[str, Dict[str, Union[int, str, List[str]]]]:
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

    def load_file(self, filename: Union[str, os.PathLike]):
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
                if line[0:len(COMMENT)] == COMMENT:
                    header_info.append(line)
                    continue
                if line[0:len(DATA_SEP)] == DATA_SEP:
                    break

                # Get the input variable name and total size for the variable
                nm: str
                sz: Union[str, int]

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

        outhdl = open(filename, 'w')
        outhdl.write(f'{self.__header}\n')
        outhdl.write(f'{HEADER_SEP*15}\n')

        for xx in self.__station_meta_header:
            outhdl.write(f'{xx}\n')

        for kk, vv in self.__input_vars_intern.items():
            if 'file_metadata' in vv:
                for xx in vv['file_metadata']:
                    outhdl.write(f'{xx}\n')

        outhdl.write(f'{HEADER_SEP*15}\n')

        for kk, vv in self.__input_vars_intern.items():
            outhdl.write(f'{kk} {vv["size"]}\n')

        outhdl.write(f'{DATA_SEP*15}\n')

        df.to_csv(outhdl, sep=' ', columns=out_order, index=False, header=False, na_rep='-999', encoding=None)
        outhdl.close()

    def _add_file_metadata(self, header_info: List[str]):
        """Add file metadata from data file.

        :param header_info: list of header lines from the data file
        """

        if self.__verbose:
            con.print(header_info)

        it = iter(header_info)
        line = next(it)

        try:
            while line[0:len(STATION_START)] != STATION_START:
                line = next(it)

            # Process the station information
            self.__station_meta_header.append(line.strip())

            line = next(it)
            if line[0:len('// ID')] == '// ID':
                # Process the station information
                self.__station_meta_header.append(line.strip())
                meta_vars = line.replace(COMMENT, '').strip().split()

                self.__df_file_metadata = pd.DataFrame(columns=meta_vars)

                line = next(it)

                # Loop through the variables and read the station information for each one
                for kk, vv in self.__input_vars_intern.items():
                    if line[0:len(HEADER_SEP)] == HEADER_SEP:
                        continue

                    for sz in range(vv['size']):
                        fields = line.replace(COMMENT, '').strip().split()
                        stn_id = fields[0]

                        dup = 0
                        while stn_id in self.__input_vars_intern[kk].setdefault('stations', []):
                            con.print(f'[orange3]WARNING[/] {kk} station {stn_id} already declared in data file; adjusting variable name')
                            dup += 1
                            stn_id = f'{stn_id}dup{dup}'

                        self.__input_vars_intern[kk].setdefault('stations', []).append(stn_id)
                        self.__input_vars_intern[kk].setdefault('file_metadata', []).append(line.strip())

                        try:
                            self.__df_file_metadata.loc[len(self.__df_file_metadata)] = fields
                        except ValueError:
                            con.print(f'[red]ERROR[/]: file metadata for {kk} station {stn_id} has incorrect number of fields')
                            raise

                        line = next(it)

                # Check that the number of stations for each variable matches the variable size
                for vv in self.__input_vars_intern.values():
                    if vv['size'] != len(vv['stations']):
                        con.print(f'[red]ERROR[/] Number of expected stations, {vv["size"]}, does not match number of stations read {vv["stations"]}')
        except StopIteration:
            con.print('[orange3]WARNING[/]: No usable metadata found')

            # We don't have useful metadata so we assign indexed station IDs
            self.__station_meta_header.append('// Station metadata:')
            self.__station_meta_header.append('// ID')

            self.__df_file_metadata = pd.DataFrame(columns=['ID', 'type'])

            for kk, vv in self.__input_vars_intern.items():
                if 'stations' not in self.__input_vars_intern[kk]:
                    for sz in range(vv['size']):
                        self.__input_vars_intern[kk].setdefault('stations', []).append(f'{sz+1}')
                        self.__input_vars_intern[kk].setdefault('file_metadata', []).append(f'{COMMENT} {sz+1}')
                        self.__df_file_metadata.loc[len(self.__df_file_metadata)] = [f'{sz+1}', kk]

        try:
            while line[0:len(UNITS_START)] != UNITS_START:
                line = next(it)

            # Process the units
            while line[0:len(HEADER_SEP)] != HEADER_SEP:
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

    def _data_column_names(self) -> List[str]:
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

    # def write_selected_stations(self, filename):
    #     """Writes station observations to a new file"""
    #     # Either writes out all station observations or, if stations are selected,
    #     # then a subset of station observations.
    #
    #     # Sample header format
    #
    #     # $Id:$
    #     # ////////////////////////////////////////////////////////////
    #     # // Station metadata (listed in the same order as the data):
    #     # // ID    Type Latitude Longitude Elevation
    #     # // <station info>
    #     # ////////////////////////////////////////////////////////////
    #     # // Unit: runoff = ft3 per sec, elevation = feet
    #     # ////////////////////////////////////////////////////////////
    #     # runoff <number of stations for each type>
    #     # ################################################################################
    #
    #     top_line = '$Id:$\n'
    #     section_sep = '////////////////////////////////////////////////////////////\n'
    #     meta_header_1 = '// Station metadata (listed in the same order as the data):\n'
    #     # metaHeader2 = '// ID    Type Latitude Longitude Elevation'
    #     meta_header_2 = '// %s\n' % ' '.join(self.metaheader)
    #     data_section = '################################################################################\n'
    #
    #     # ----------------------------------
    #     # Get the station information for each selected station
    #     type_count = {}  # Counts the number of stations for each type of data (e.g. 'runoff')
    #     stninfo = ''
    #     if self.__selectedStations is None:
    #         for xx in self.__stations:
    #             if xx[1] not in type_count:
    #                 # index 1 should be the type field
    #                 type_count[xx[1]] = 0
    #             type_count[xx[1]] += 1
    #
    #             stninfo += '// %s\n' % ' '.join(xx)
    #     else:
    #         for xx in self.__selectedStations:
    #             cstn = self.__stations[self.__stationIndex[xx]]
    #
    #             if cstn[1] not in type_count:
    #                 # index 1 should be the type field
    #                 type_count[cstn[1]] = 0
    #
    #             type_count[cstn[1]] += 1
    #
    #             stninfo += '// %s\n' % ' '.join(cstn)
    #     # stninfo = stninfo.rstrip('\n')
    #
    #     # ----------------------------------
    #     # Get the units information
    #     unit_line = '// Unit:'
    #     for uu in self.__units:
    #         unit_line += ' %s,' % ' = '.join(uu)
    #     unit_line = '%s\n' % unit_line.rstrip(',')
    #
    #     # ----------------------------------
    #     # Create the list of types of data that are being included
    #     tmpl = []
    #
    #     # Create list of types in the correct order
    #     for (kk, vv) in self.__types.items():
    #         if kk in type_count:
    #             tmpl.insert(vv[0], [kk, type_count[kk]])
    #
    #     type_line = ''
    #     for tt in tmpl:
    #         type_line += '%s %d\n' % (tt[0], tt[1])
    #     # typeLine = typeLine.rstrip('\n')
    #
    #     # Write out the header to the new file
    #     outfile = open(filename, 'w')
    #     outfile.write(top_line)
    #     outfile.write(section_sep)
    #     outfile.write(meta_header_1)
    #     outfile.write(meta_header_2)
    #     outfile.write(stninfo)
    #     outfile.write(section_sep)
    #     outfile.write(unit_line)
    #     outfile.write(section_sep)
    #     outfile.write(type_line)
    #     outfile.write(data_section)
    #
    #     # Write out the data to the new file
    #     # Using quoting=csv.QUOTE_NONE results in an error when using a customized  date_format
    #     # A kludgy work around is to write with quoting and then re-open the file
    #     # and write it back out, stripping the quote characters.
    #     self.data.to_csv(outfile, index=True, header=False, date_format='%Y %m %d %H %M %S', sep=' ')
    #     outfile.close()
    #
    #     old = open(filename, 'r').read()
    #     new = re.sub('["]', '', old)
    #     open(filename, 'w').write(new)
    #
    #     # def getRecurrenceInterval(self, thetype):
    #     #     """Returns the recurrence intervals for each station"""
    #     #
    #     #     # Copy the subset of data
    #     #     xx = self.seldata(thetype)
    #     #
    #     #     ri = np.zeros(xx.shape)
    #     #     ri[:,:] = -1.
    #     #
    #     #     # for each station we need to compute the RI for non-zero values
    #     #     for ss in range(0,xx.shape[1]):
    #     #         tmp = xx[:,ss]              # copy values for current station
    #     #
    #     #         # Get array of indices that would result in a sorted array
    #     #         sorted_ind = np.argsort(tmp)
    #     #         #print "sorted_ind.shape:", sorted_ind.shape
    #     #
    #     #         numobs = tmp[(tmp > 0.0),].shape[0]  # Number of observations > 0.
    #     #         nyr = float(numobs / 365)     # Number of years of non-zero observations
    #     #
    #     #         nz_cnt = 0  # non-zero value counter
    #     #         for si in sorted_ind:
    #     #             if tmp[si] > 0.:
    #     #                 nz_cnt += 1
    #     #                 rank = numobs - nz_cnt + 1
    #     #                 ri[si,ss] = (nyr + 1.) / float(rank)
    #     #                 #print "%s: [%d]: %d %d %0.3f %0.3f" % (ss, si,  numobs, rank, tmp[si], ri[si,ss])
    #     #
    #     #     return ri
# ***** END of class streamflow()
