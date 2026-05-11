from __future__ import annotations

import fsspec
import numpy as np
import pandas as pd   # type: ignore
import netCDF4 as nc   # type: ignore
import xarray as xr   # type: ignore

from collections import defaultdict
from pathlib import Path

from ..control.Control import Control
from ..constants import MetaDataType, NEW_PTYPE_TO_DTYPE
from ..base.console import get_console_instance
from ..parameters.Parameters import Parameters

con = None

__author__ = 'Parker Norton (pnorton@usgs.gov)'

NA_VALS_DEFAULT = ('-99.0', '-999.0', 'NaN', 'inf')
DATA_SEP = '####'

# Crosswalk of some of the possible source CBH variable names to PRMS variable names
var_crosswalk: dict[str, str] = dict(tmax='tmax_hru',
                                     T2MAX='tmax_hru',
                                     tmin='tmin_hru',
                                     T2MIN='tmin_hru',
                                     precip='hru_ppt',
                                     prcp='hru_ppt',
                                     RAIN='hru_ppt',
                                     rhavg='humidity_hru')
temp_units = {0: 'degree_fahrenheit', 1: 'degree_celsius'}
precip_units = {0: 'inch', 1: 'mm'}


class Cbh(object):
    """Climate-By-HRU (CBH) files for PRMS."""

    def __init__(self, src_path: str | Path | list[str | Path],
                 metadata: MetaDataType,
                 engine: str = 'ascii',
                 control: Control | None = None,
                 parameters: Parameters | None = None,
                 verbose: bool = False):
        """
        :param src_path: List of paths to CBH files
        :param metadata: Metadata dictionary for Climate-by-HRU variables
        :param engine: Engine to use for reading CBH files (one of netcdf, zarr, or ascii)
        :param control: Control object for PRMS model containing configuration information
        :param verbose: Output debugging information
        """

        global con
        con = get_console_instance()

        self.verbose = verbose
        self.has_nhm_id = False
        self.metadata = metadata['cbh']
        self.__parameters = parameters
        self.__var_map = {}
        self.__cbh_src: dict[str, str] = {}

        if isinstance(src_path, str):
            src_path = Path(src_path)

        if isinstance(src_path, list):
            self.__src_path = [Path(ff).resolve() for ff in src_path]
        else:
            if '*' in src_path.name:
                # wildcard character in filename so glob the path
                self.__src_path = list(src_path.parent.glob(src_path.name))
            else:
                self.__src_path = [src_path.resolve()]

        # con.print(f'CBH files: {self.__src_path}')

        if parameters is not None:
            self.resolve_units()

        assert self.__src_path is not None

        match engine:
            case 'netcdf':
                ds = xr.open_mfdataset(self.__src_path, chunks={}, combine='by_coords',
                                        compat='no_conflicts', join='outer',
                                       data_vars='minimal', decode_cf=True, engine='netcdf4',
                                       parallel=False)
            case 'zarr':
                if len(self.__src_path) > 1:
                    con.print('[red]ERROR[/]: Zarr engine does not support reading multiple files')
                elif not self.__src_path[0].is_dir():
                    con.print('[red]ERROR[/]: Zarr engine requires a directory of files')
                else:
                    ds = xr.open_zarr(self.__src_path[0], consolidated=True)
            case 'ascii':
                cbh_files = {}
                if control is None:
                    con.print('[orange3]WARNING[/]: No control object provided; CBH variables may be missing metadata')

                    for kk in self.__src_path:
                        cbh_files[kk] = None
                    ds = self._cbh_to_xarray(cbh_files)  # type: ignore
                else:
                    # When a control object is specified, the src_path indicates
                    # the model directory and the *_day variables are read to get
                    # candidate CBH files.
                    cbh_file_vars = dict(albedo_day='albedo_hru',
                                         cloud_cover_day='cloud_cover_cbh',
                                         humidity_day='humidity_hru',
                                         potet_day='potet',
                                         precip_day='hru_ppt',
                                         swrad_day='swrad',
                                         tmax_day='tmax_hru',
                                         tmin_day='tmin_hru',
                                         transp_day='transp_on',
                                         windspeed_day='windspeed_hru')

                    for ctl_var, prms_var in cbh_file_vars.items():
                        # Get the filename associated with the control variable
                        cfile = control.get(ctl_var).values
                        assert type(cfile) is str

                        if (self.__src_path[0] / cfile).exists():
                            if self.verbose:
                                con.print(f'[green]INFO[/]: Found {cfile}')
                            cbh_files[self.__src_path[0] / cfile] = prms_var

                    ds = self._cbh_to_xarray(cbh_files)  # type: ignore # , variables=cbh_vars)

        if 'nhm_id' in ds.data_vars:
            # The dataset has nhm_id variable so use it as the nhru dimension
            ds = ds.assign_coords(nhru=ds.nhm_id)
            self.has_nhm_id = True

        for cvar in ds.data_vars:
            self.__var_map[str(cvar)] = var_crosswalk.get(str(cvar), str(cvar))

        self.__dataset = ds

    @property
    def data(self) -> xr.Dataset:
        """Returns the CBH dataset.

        :returns: xarray dataset
        """

        return self.__dataset

    @property
    def var_map(self) -> dict[str, str]:
        """Return variable-to-prms_variable mapping."""

        return self.__var_map

    @property
    def cbh_src(self) -> dict[str, str]:
        """Return variable to source-file mapping."""

        return self.__cbh_src

    def resolve_units(self):
        """Adjust units metadata for CBH variables that have an initial units value of
        precip_units or temp_units.

        :returns: None
        """

        selected_units = self.__parameters.user_defined_units
        for cvar, cval in self.metadata.items():
            if cval['units'] in selected_units:
                cval['units'] = selected_units[cval['units']]

    def set_nhm_id(self, nhm_ids: np.ndarray):
        """Add the model nhm_id parameter as a coordinate variable.

        :param nhm_ids: array of nhm_id values
        """

        # Returns ValueError if the nhm_ids size does not match the dataset dimension size
        if not self.has_nhm_id:
            # Only add nhm_id if it doesn't already exist
            self.__dataset['nhm_id'] = ('nhru', nhm_ids)
            self.__dataset['nhm_id'].attrs['long_name'] = 'Global model Hydrologic Response Unit ID (HRU)'
            self.__dataset = self.__dataset.assign_coords(nhru=self.__dataset.nhm_id)
            self.has_nhm_id = True

    def write_ascii(self, filename: str | Path,
                    variable: str,
                    time_slice: list | slice | None = None,
                    hru_ids: list | np.ndarray | None = None,
                    na_rep: str = '-999',
                    float_format: str = '%0.2f'):
        """Write CBH data for selected variable to PRMS ASCII-formatted file.

        :param filename: Climate-by-HRU filename
        :param variable: CBH variable to write
        :param time_slice: time slice to write
        :param hru_ids: list or array of HRU IDs (local IDs if has_nhm_id is false) to write
        """

        if hru_ids is None:
            # Return all HRUs if hru_ids is not provided
            hru_ids = self.__dataset['nhru'].values

        if time_slice is None:
            # Return all time steps if time_slice is not provided
            time_slice = slice(self.__dataset['time'][0].values,
                               self.__dataset['time'][-1].values)

        if isinstance(time_slice, list):
            time_slice = slice(time_slice[0], time_slice[-1])

        # For out_order the first six columns contain the time information and
        # are always output for the cbh files
        # out_order: List[Union[int, str]] = [kk for kk in self.__nhm_hrus]
        out_order = [kk for kk in hru_ids]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        # variable = var_crosswalk.get(variable, variable)

        if variable in self.__dataset.data_vars:
            ds = self.__dataset[variable].sel(nhru=hru_ids, time=time_slice).to_pandas()

            # Add time information as columns
            ds['year'] = ds.index.year
            ds['month'] = ds.index.month
            ds['day'] = ds.index.day
            ds['hour'] = 0
            ds['minute'] = 0
            ds['second'] = 0

            out_cbh = open(filename, 'w')
            out_cbh.write('Written by Bandit\n')
            out_cbh.write(f'{var_crosswalk.get(variable, variable)} {len(hru_ids)}\n')
            out_cbh.write('########################################\n')
            ds.to_csv(out_cbh, columns=out_order, na_rep=na_rep, float_format=float_format,
                      sep=' ', index=False, header=False, lineterminator='\n', encoding=None,
                      chunksize=10)
            out_cbh.close()
        else:
            print(f'WARNING: {variable} does not exist in source CBH files..skipping')

    def write_netcdf(self, filename: str | Path,
                     variables: list[str] | None = None,
                     global_attrs: dict | None = None,
                     time_slice: list | slice | None = None,
                     hru_ids: list | np.ndarray | None = None):
        """Write CBH variables to netCDF file.

        :param filename: name of netCDF output file
        :param variables: list of CBH variables to write
        :param global_attrs: optional dictionary of global attributes to include in netcdf file
        :param time_slice: time slice to write
        :param hru_ids: list or array of HRU IDs (local IDs if has_nhm_id is false) to write
        """

        if hru_ids is None:
            # Return all HRUs if hru_ids is not provided
            hru_ids = self.__dataset['nhru'].values

        if time_slice is None:
            # Return all time steps if time_slice is not provided
            time_slice = slice(self.__dataset['time'][0].values,
                               self.__dataset['time'][-1].values)

        if isinstance(time_slice, list):
            time_slice = slice(time_slice[0], time_slice[-1])

        ds = self.__dataset.sel(nhru=hru_ids, time=time_slice)
        # ds = ds.sel(time=slice(self.__stdate, self.__endate), nhru=self.__nhm_hrus)

        if variables is None:
            pass
        elif isinstance(variables, list):
            ds = ds[variables]

        # Remove _FillValue from coordinate variables
        for vv in list(ds.coords):
            ds[vv].encoding.update({'_FillValue': None})

        if 'crs' in ds.variables:
            # ds['crs'] = self.__dataset['crs']
            ds['crs'].encoding.update({'_FillValue': None,
                                       'contiguous': True})

        ds['time'].attrs['standard_name'] = 'time'
        ds['time'].attrs['long_name'] = 'time'

        # Add nhm_id variable which will be the global NHM IDs
        ds['nhm_id'] = ds['nhru']
        ds['nhm_id'].attrs['long_name'] = 'Global model Hydrologic Response Unit ID (HRU)'

        if not self.has_nhm_id:
            ds['nhm_id'].attrs['note'] = 'Locally generated ID; not the same as NHM global ID'

        # Change the nhru coordinate variable values to reflect the local model HRU IDs
        ds = ds.assign_coords(nhru=np.arange(1, ds.nhru.values.size+1, dtype=ds.nhru.dtype))
        ds['nhru'].attrs['long_name'] = 'Local model Hydrologic Response Unit ID (HRU)'
        ds['nhru'].attrs['cf_role'] = 'timeseries_id'

        # Add/update global attributes
        ds.attrs['Description'] = 'Climate-by-HRU'

        if global_attrs is not None:
            for kk, vv in global_attrs.items():
                ds.attrs[kk] = vv

        encoding = {}

        for cvar in ds.variables:
            if ds[cvar].ndim > 1:
                try:
                    encoding[cvar] = dict(_FillValue=ds[cvar].encoding['_FillValue'],
                                          compression='zlib',
                                          complevel=2,
                                          fletcher32=True)
                except KeyError:
                    con.print(f'[orange3]WARNING[/]: Variable {cvar} has no _FillValue defined in encoding.')
                    encoding[cvar] = dict(_FillValue=None,
                                          compression='zlib',
                                          complevel=2,
                                          fletcher32=True)
            else:
                encoding[cvar] = dict(_FillValue=None,
                                      contiguous=True)

        ds.load().to_netcdf(filename, engine='netcdf4', format='NETCDF4', encoding=encoding)

    def _cbh_to_xarray(self, filename: dict[Path, str | None]) -> xr.Dataset:
        # variables: list[str] | None = None) -> xr.Dataset:
        """Convert ASCII CBH file(s) to xarray

        :param filename: list of CBH filepaths or a single CBH filename
        :returns: xarray dataset of CBH data
        """

        var_meta = dict(time=dict(standard_name='time', long_name='time'),
                        nhru=dict(standard_name='nhru', long_name='Local model Hydrologic Response Unit ID (HRU)'))

        var_enc = {'float32': dict(_FillValue=nc.default_fillvals['f4']),
                   'float64': dict(_FillValue=nc.default_fillvals['f8']),
                   'int32': dict(_FillValue=nc.default_fillvals['i4']),
                   'int64': dict(_FillValue=nc.default_fillvals['i8'])}

        df = []

        for cfile, cvar in filename.items():
            read_ok = True

            if isinstance(cfile, str):
                cfile = Path(cfile)
                # assert isinstance(cfile, Path)

            # First get the header info which has the variable name and number of HRUs
            with open(cfile, 'r') as fhdl:
                # First line is a descriptive header
                header = fhdl.readline().rstrip()

                # Next line has the variable name (not checked by PRMS) and number of dimensions
                var_name, ndims = fhdl.readline().rstrip().split()
                ndims = int(ndims)   # type: ignore

                if cvar is not None:
                    if var_name != cvar:
                        con.print(f'[orange3]WARNING[/]: Variable name in file header ({var_name}) does not match expected '
                                  f'variable ({cvar}). The expected variable name will be used.')
                        var_name = cvar
                else:
                    # With ASCII files when cvar is None, usually the control object was not provided.
                    # Try looking up the variable name in the variable crosswalk.
                    var_name = var_crosswalk.get(var_name, var_name)

                self.__cbh_src[var_name] = cfile.name

                line = fhdl.readline().rstrip()

                if line[0:len(DATA_SEP)] != DATA_SEP:
                    if line.split()[0] == 'orad':
                        # This happens when orad_flag == 1
                        con.print(f'[red]ERROR[/]: Two variables in CBH file ({var_name}, orad). Data will not be read.')
                    else:
                        con.print(f'[red]ERROR[/]: {cvar}: Unknown extra line: {line}.\n Data will not be read.')

                    read_ok = False

            if read_ok:
                df.append(pd.DataFrame(self._read_ascii_file(cfile,
                                                             datatype=NEW_PTYPE_TO_DTYPE[self.metadata[var_name]['datatype']]).stack()))
                df[-1].index.rename(['time', 'nhru'], inplace=True)
                # df[-1].rename(var_name, inplace=True)
                df[-1].rename(columns={0: var_name}, inplace=True)

        ds = xr.merge([cdf.to_xarray() for cdf in df])

        # Apply metadata to variables
        for cvar in ds.variables:
            if cvar in self.metadata:
                cattrs = self.metadata[cvar]

                ds[cvar] = ds[cvar].astype(NEW_PTYPE_TO_DTYPE[cattrs['datatype']])

                ds[cvar].attrs['long_name'] = cattrs['description']

                if cattrs['units'] == 'temp_units':
                    # For now just default to degrees_fahrenheit
                    ds[cvar].attrs['units'] = temp_units[0]
                elif cattrs['units'] == 'precip_units':
                    # For now just default to inches
                    ds[cvar].attrs['units'] = precip_units[0]
                else:
                    ds[cvar].attrs['units'] = cattrs['units']

                # Set the fill value
                # con.print(f'{cvar}: {cattrs["datatype"]} -> {var_enc[cattrs["datatype"]]}')
                ds[cvar].encoding.update(var_enc[cattrs['datatype']])
            else:
                if cvar not in ['time', 'nhru']:
                    con.print(f'[orange3]WARNING[/]: {cvar} not found in metadata.')

            if cvar in var_meta:
                for cattr, cval in var_meta[cvar].items():   # type: ignore
                    ds[cvar].attrs[cattr] = cval

        return ds

    @staticmethod
    def _read_ascii_file(filename: str | Path,
                         datatype=np.float32,
                         columns: list | None = None) -> pd.DataFrame:
        """Reads a single ASCII CBH file.

        :param filename: name of the CBH file
        :param datatype: datatype to use for reading the CBH variable
        :param columns: columns to read
        :returns: dataframe of CBH variable
        """

        if isinstance(filename, str):
            filename = Path(filename)

        # Written by Bandit
        # prcp 14
        # ########################################
        # 1980 1 1 0

        # Columns 0-5 always represent date/time information
        time_col_names = {0: 'year', 1: 'month', 2: 'day', 3: 'hour', 4: 'minute', 5: 'second'}

        types = defaultdict(lambda: datatype)
        types[0] = np.int32
        types[1] = np.int32
        types[2] = np.int32
        types[3] = np.int32
        types[4] = np.int32
        types[5] = np.int32

        df = pd.read_csv(filename, sep=' ', skipinitialspace=True,
                         skiprows=3, engine='c', dtype=types, low_memory=True,
                         # skiprows=3, engine='c', memory_map=True,
                         header=None, na_values=NA_VALS_DEFAULT,
                         usecols=columns)

        df[0] = pd.to_numeric(df[0], downcast='integer')
        df[1] = pd.to_numeric(df[1], downcast='integer')
        df[2] = pd.to_numeric(df[2], downcast='integer')
        df[3] = pd.to_numeric(df[3], downcast='integer')
        df[4] = pd.to_numeric(df[4], downcast='integer')
        df[5] = pd.to_numeric(df[5], downcast='integer')

        # Rename columns with time information
        df.rename(columns=time_col_names, inplace=True)

        # Rename columns with local model indices
        ren_dict = {k + 6: k + 1 for k in range(len(df.columns))}
        df.rename(columns=ren_dict, inplace=True)

        df['time'] = pd.to_datetime(df[time_col_names.values()], yearfirst=True)
        df.drop(columns=time_col_names.values(), inplace=True)
        df.set_index('time', inplace=True)

        return df
