import datetime
import fsspec   # type: ignore
import numpy as np
import os
import pandas as pd   # type: ignore
import netCDF4 as nc   # type: ignore
import xarray as xr

from pathlib import Path
from typing import Dict, List, Optional, Union

__author__ = 'Parker Norton (pnorton@usgs.gov)'

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class CbhNetcdf(object):
    """Climate-By-HRU (CBH) files in netCDF format."""

    def __init__(self, src_path: Union[str, Path],
                 nhm_hrus: List[int],
                 st_date: Optional[datetime.datetime] = None,
                 en_date: Optional[datetime.datetime] = None):
        """
        :param src_path: Full path to netCDF file
        :param st_date: The starting date for restricting CBH results
        :param en_date: The ending date for restricting CBH results
        :param nhm_hrus: List of NHM HRU IDs to extract from CBH
        """

        if isinstance(src_path, str):
            src_path = Path(src_path)

        self.__src_path = src_path
        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus

        self.__dataset = self.read_netcdf()

    @property
    def data(self) -> xr.Dataset:
        """Returns the CBH dataset.

        :returns: xarray dataset
        """

        return self.__dataset

    def read_netcdf(self) -> xr.Dataset:
        """Read CBH files stored in netCDF format.

        :returns: xarray dataset
        """

        filepath = str(self.__src_path.resolve())
        match self.__src_path.suffix:
            case '.nc':
                ds = xr.open_mfdataset(filepath, chunks={}, combine='by_coords',
                                       data_vars='minimal', decode_cf=True, engine='netcdf4',
                                       parallel=True)
            case '.zarr':
                ds = xr.open_zarr(filepath, consolidated=True)
            case '.json':
                fs = fsspec.filesystem('reference', fo=filepath)
                m = fs.get_mapper('')

                ds = xr.open_dataset(m, engine='zarr', chunks={}, backend_kwargs={'consolidated': False})

        if 'nhm_id' in ds.data_vars:
            # dataset has nhm_id variable so use it as the nhru dimension
            ds = ds.assign_coords(nhru=ds.nhm_id)

        if self.__stdate is None:
            # Use first date from the dataset
            self.__stdate = pd.to_datetime(str(ds.time[0].values))

        if self.__endate is None:
            # Use last date from the dataset
            self.__endate = pd.to_datetime(str(ds.time[-1].values))

        return ds

    def get_var(self, var: str) -> pd.DataFrame:
        """Get a variable from the netCDF file.

        :param var: Name of the variable
        :returns: dataframe of variable values
        """

        try:
            data = self.__dataset[var].sel(time=slice(self.__stdate, self.__endate),
                                           nhru=self.__nhm_hrus).to_pandas()
        except IndexError:
            print(f'ERROR: Dimensions (time, nhru) were used to subset {var} which expects' +
                  f'dimensions ({" ".join(map(str, self.__dataset[var].coords))})')
            raise
        except KeyError:
            # Happens when older hruid dimension is used instead of nhru
            data = self.__dataset[var].sel(time=slice(self.__stdate, self.__endate),
                                           hruid=self.__nhm_hrus).to_pandas()

        return data

    def write_ascii(self, filename: Union[str, os.PathLike, Path],
                    variable: str):
        """Write CBH data for variable to PRMS ASCII-formatted file.

        :param filename: Climate-by-HRU filename
        :param variable: CBH variable to write
        """

        # For out_order the first six columns contain the time information and
        # are always output for the cbh files
        out_order: List[Union[int, str]] = [kk for kk in self.__nhm_hrus]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        if variable in self.__dataset.data_vars:
            data = self.get_var(var=variable)

            # Add time information as columns
            data['year'] = data.index.year
            data['month'] = data.index.month
            data['day'] = data.index.day
            data['hour'] = 0
            data['minute'] = 0
            data['second'] = 0

            out_cbh = open(filename, 'w')
            out_cbh.write('Written by Bandit\n')
            out_cbh.write(f'{variable} {len(self.__nhm_hrus)}\n')
            out_cbh.write('########################################\n')
            # data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.3f',
            data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.2f',
                        sep=' ', index=False, header=False, lineterminator='\n', encoding=None, chunksize=50)
            out_cbh.close()
        else:
            print(f'WARNING: {variable} does not exist in source CBH files..skipping')

    def write_netcdf(self, filename: Union[str, os.PathLike, Path],
                     variables: Optional[List[str]] = None,
                     global_attrs: Optional[Dict] = None):
        """Write CBH variables to netCDF file.

        :param filename: name of netCDF output file
        :param variables: list of CBH variables to write
        :param global_attrs: optional dictionary of attributes to include in netcdf file
        """

        ds = self.__dataset
        ds = ds.sel(time=slice(self.__stdate, self.__endate), nhru=self.__nhm_hrus)

        if variables is None:
            pass
        elif isinstance(variables, list):
            ds = ds[variables]

        # Remove _FillValue from coordinate variables
        for vv in list(ds.coords):
            ds[vv].encoding.update({'_FillValue': None})

        ds['crs'] = self.__dataset['crs']
        ds['crs'].encoding.update({'_FillValue': None,
                                   'contiguous': True})

        ds['time'].attrs['standard_name'] = 'time'
        ds['time'].attrs['long_name'] = 'time'

        # Add nhm_id variable which will be the global NHM IDs
        ds['nhm_id'] = ds['nhru']
        ds['nhm_id'].attrs['long_name'] = 'Global model Hydrologic Response Unit ID (HRU)'

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
                encoding[cvar] = dict(_FillValue=ds[cvar].encoding['_FillValue'],
                                      compression='zlib',
                                      complevel=2,
                                      fletcher32=True)
            else:
                encoding[cvar] = dict(_FillValue=None,
                                      contiguous=True)

        ds.load().to_netcdf(filename, engine='netcdf4', format='NETCDF4', encoding=encoding)
