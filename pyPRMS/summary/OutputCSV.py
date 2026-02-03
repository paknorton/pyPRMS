import pandas as pd   # type: ignore

from pathlib import Path
from typing import Optional, Union

from ..base.console import get_console_instance

con = None


class OutputCSV(object):
    """Class for working with PRMS CSV output files.
    """

    def __init__(self, filename: Union[str, Path],
                 verbose: Optional[bool] = False):
        """Initialize the OutputCSV object.

        :param filename: Name of the PRMS CSV output file
        :param verbose: Output debugging information
        """

        global con
        con = get_console_instance()

        self.__filename = filename

        if isinstance(self.__filename, str):
            self.__filename = Path(self.__filename)

        self.verbose = verbose
        self.__data = None

        self.__pois = []
        self.__poi_segments = {}
        self.__basin_vars = []
        self.__col_var = {}

        self._read_csv_header()
        self._read_csv_ascii()

    @property
    def basin_vars(self):
        """Returns the basin variables from the CSV output file."""
        return self.__basin_vars

    @property
    def data(self):
        return self.__data

    @property
    def pois(self):
        return self.__pois

    @property
    def poi_segments(self):
        return self.__poi_segments

    @property
    def variables(self):
        return sorted(list(self.__col_var.values()))

    def _read_csv_header(self):
        """Read the headers from a PRMS CSV model output file"""

        fhdl = open(self.__filename, 'r')

        # First row contains field names
        # Second row is a a mix of field names (for the date) and data types
        hdr1 = fhdl.readline().strip()
        hdr2 = fhdl.readline().strip()
        fhdl.close()

        # Determine the value separator
        # Check for comma first; some files have commas and spaces
        self.sep = ' '

        match ',' in hdr1:
            case True:
                self.sep = ','

        if self.verbose:
            con.print(f'[green]INFO[/]: value separator = {self.sep}')

        # Determine the format of the time field(s)
        var_offset = 0
        if 'year month day' in hdr2:
            var_offset = 3
            self.time_col_names = {0: 'year', 1: 'month', 2: 'day'}
        elif 'year-month-day' in hdr2:
            var_offset = 1
            self.time_col_names = {0: 'Date'}

        if self.verbose:
            con.print(f'[green]INFO[/]: {var_offset=}')
            con.print(f'[green]INFO[/]: time field(s): {list(self.time_col_names.values())}')

        # Parse the field names
        tmp_flds = [kk.strip() for kk in hdr1.split(self.sep)]
        tmp_flds.remove('Date')

        flds = {nn+var_offset: hh for nn, hh in enumerate(tmp_flds)}

        for xx, yy in flds.items():
            if 'seg_outflow' in yy:
                tfld = yy.split('_')
                segid = int(tfld[2]) - 1  # Change to zero-based indices
                poiid = tfld[4]

                self.__col_var[xx] = poiid
                self.__pois.append(poiid)
                self.__poi_segments[poiid] = segid
            else:
                self.__col_var[xx] = yy.strip()
                self.__basin_vars.append(yy.strip())

    def _read_csv_ascii(self):
        """Read a PRMS CSV model output file
        """

        poi_idx = {vv: kk for kk, vv in self.__col_var.items()}
        sel_poi_idx = [poi_idx[kk] for kk in self.variables]

        sel_cols = list(self.time_col_names.keys())
        sel_cols.extend(sel_poi_idx)

        # Read the CSV without the included field names
        df = pd.read_csv(self.__filename, sep=self.sep, skipinitialspace=True,
                         header=None, skiprows=2, usecols=sel_cols)

        df.rename(columns=self.time_col_names, inplace=True)

        df.rename(columns=self.__col_var, inplace=True)

        if len(self.time_col_names) > 1:
            df['time'] = pd.to_datetime(df[self.time_col_names.values()], yearfirst=True)
            df.drop(columns=self.time_col_names.values(), inplace=True)
        else:
            df.rename(columns={self.time_col_names[0]: 'time'}, inplace=True)

        df.set_index('time', inplace=True)

        self.__data = df
