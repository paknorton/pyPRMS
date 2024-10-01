import os
import pandas as pd   # type: ignore

from rich.console import Console
from rich import pretty

from typing import Dict, List, Optional, Sequence, Tuple, Union
pretty.install()
con = Console()

# TS_FORMAT = '%Y %m %d %H %M %S'   # 1915 1 13 0 0 0

HEADER_SEP = '//////////'
STATION_START = '// Station IDs for'
UNITS_START = '// Unit:'
DATA_SEP = '####'
COMMENT = '//'


class DataFile(object):
    """Class for working with observed streamflow in the PRMS ASCII data file format"""

    def __init__(self, filename: Union[str, os.PathLike],
                 missing: Sequence[str] = ('-99.9', '-999.0', '-9999.0'),
                 verbose: bool = False,
                 include_metadata: bool = True):

        self.__missing = missing
        self.filename = filename
        self.__verbose = verbose
        self.__include_metadata = include_metadata

        self.__timecols = 6  # number columns for time in the file
        self.__header = ''   # data file header from first line of the file

        # Dictionary of input variables and associated metadata
        self.__input_vars: Dict[str, Dict[str, Union[int, str, List[str], pd.DataFrame]]] = {}
        self.__data_raw: Optional[pd.DataFrame] = None

        self.load_file(self.filename)

    @property
    def data(self):
        """Pandas dataframe of the station data for each input variable"""

        return self.__data_raw

    @property
    def input_variables(self):
        """Get the input variables in the data file"""

        return self.__input_vars

    def data_by_variable(self, variable: str) -> pd.DataFrame:
        """Get the data for a specific input variable"""

        assert type(self.__input_vars[variable]['data']) is pd.DataFrame
        return self.__input_vars[variable]['data']

    def get(self, name: str) -> Dict:
        """Get the metadata for a specific input variable"""

        return self.__input_vars[name]

    def load_file(self, filename: Union[str, os.PathLike]):
        """Read the PRMS ASCII streamflow data file"""

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
                    if nm in self.__input_vars:
                        raise KeyError(f'{nm} declared multiple times in the data file')
                    self.__input_vars[nm] = dict(size=sz)

            # =============================
            # Process metadata
            self._add_metadata(header_info)

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

    def _add_metadata(self, header_info):
        """Add metadata from data file
        """

        it = iter(header_info)
        if 'Downsizer' in self.__header or 'Bandit' in self.__header:
            for line in it:
                if line[0:len(STATION_START)] == STATION_START:
                    # Process the station information
                    station_vars = line[len(STATION_START):].replace(' ', '').strip(':').split(',')

                    line = next(it)
                    if line == '// ID':
                        # Skip the metadata column information line (found Bandit data files)
                        line = next(it)

                    while line[0:len(HEADER_SEP)] != HEADER_SEP and len(line) > 2:
                        for cvar in station_vars:
                            if cvar not in self.__input_vars:
                                raise KeyError(f'{cvar} is not one of the input variables declared in the data file')
                            self.__input_vars[cvar].setdefault('stations', []).extend(line.replace(COMMENT, '').
                                                                                      replace(' ', '').split(','))
                        line = next(it)
                elif line[0:len(UNITS_START)] == UNITS_START:
                    # Process the units
                    while line[0:len(HEADER_SEP)] != HEADER_SEP:
                        for elem in line.replace(UNITS_START, '').replace(COMMENT, '').replace(' ', '').split(','):
                            cvar, cunits = elem.split('=')
                            try:
                                self.__input_vars[cvar]['units'] = cunits
                            except KeyError:
                                con.print(f'[red]{cvar}[/] is not a valid input variable name in this data file')
                                pass
                        line = next(it)

    def _add_variable_data(self):
        """Add data to each input variable
        """

        # Create a data key for each input variable that maps to their respective parts of the dataframe
        st_idx = 0
        for cvar, cmeta in self.__input_vars.items():
            self.__input_vars[cvar]['data'] = self.__data_raw.iloc[:, st_idx:(st_idx+cmeta['size'])]
            st_idx += cmeta['size']

    def _data_column_names(self):
        """Create column names for the dataframe
        """
        var_col_names = []

        for cvar, meta in self.__input_vars.items():
            if 'stations' in meta:
                for cstn in meta['stations']:
                    var_col_names.append(f'{cvar}_{cstn}')
            else:
                # No usable metadata in the data file
                for idx in range(1, meta['size']+1):
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
