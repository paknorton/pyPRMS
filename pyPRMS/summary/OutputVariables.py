import os
import pandas as pd   # type: ignore
import xarray as xr

from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

# os.environ['USE_PYGEOS'] = '0'
# import geopandas   # type: ignore
# import cartopy.crs as ccrs  # type: ignore
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # type: ignore
# import matplotlib as mpl        # type: ignore
# import matplotlib.pyplot as plt     # type: ignore

from ..constants import MetaDataType
from ..control.Control import Control
from .OutputVariable import OutputVariable
# from ..plot_helpers import (set_colormap, get_projection, plot_line_collection, plot_polygon_collection,
#                             get_figsize, read_gis)


class OutputVariables(object):
    def __init__(self,
                 control: Control,
                 metadata: MetaDataType,
                 model_dir: Optional[Union[str, os.PathLike, Path]] = None,
                 verbose: Optional[bool] = False):
        """Initialize the model output object.

        The OutputVariables class reads ASCII model output files based on the
        output variables defined in a model control file.

        :param control: Control object
        :param metadata: Metadata for the model output variables
        :param model_dir: PRMS Model directory
        :param verbose: Output debugging information
        """

        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.__model_dir = model_dir

        self.__control = control
        self.metadata = metadata['variables']
        self.__dimension_metadata = metadata['dimensions']
        self.__verbose = verbose

        self.__out_vars = dict()

        self.__hru_poly = None
        self.__hru_shape_key: Optional[str] = None
        self.__seg_poly = None
        self.__seg_shape_key: Optional[str] = None

        for cvar, cfile in self.available_vars.items():
            self.__out_vars[cvar] = OutputVariable(cvar, cfile, self.metadata)

    @cached_property
    def available_vars(self) -> Dict[str, str]:
        """Returns dictionary of available variables and file paths

        :returns: Dictionary of available variables and file paths
        """

        var_dict = {}

        # Available options for *OutON_OFF variables in control file
        var_kind = dict(nhru=[1, 2],
                        nsegment=[1, 2],
                        nsub=[1])

        for ckind, cvals in var_kind.items():
            if self.__control.get(f'{ckind}OutON_OFF').values in cvals:
                prefix = self.__control.get(f'{ckind}OutBaseFileName').values
                varlist = self.__control.get(f'{ckind}OutVar_names').values.tolist()

                if self.__model_dir:
                    prefix = f'{self.__model_dir}/{prefix}'

                for vv in varlist:
                    var_dict[vv] = f'{prefix}{vv}.csv'

                    if ckind in ['nhru', 'nsegment']:
                        # Option 2 outputs nhm_id or nhm_seg IDs for the header instead of local model IDs
                        self.metadata[vv]['is_global'] = self.__control.get(f'{ckind}OutON_OFF').values == 2

        if self.__control.get('basinOutON_OFF').values == 1:
            filename = self.__control.get('basinOutBaseFileName').values
            varlist = self.__control.get('basinOutVar_names').values.tolist()

            if self.__model_dir:
                filename = f'{self.__model_dir}/{filename}'

            for vv in varlist:
                var_dict[vv] = f'{filename}.csv'

        return var_dict

    def get(self, varname: str) -> OutputVariable:
        """Get output variable object.

        :param varname: Name of output variable
        :returns: OutputVariable object
        """

        return self.__out_vars[varname]

    def write_netcdf(self, filename: Union[str, os.PathLike],
                     varnames: Union[str, List[str]]):
        """Write selected output variables to netCDF file.

        :param filename: Name of the netCDF file
        :param varnames: List of output variable names to write
        """

        if isinstance(varnames, str):
            varnames = [varnames]

        arr_list = []

        for cvar in varnames:
            arr_list.append(self.__out_vars[cvar].to_xarray())

        ds = xr.merge(arr_list)
        ds.to_netcdf(filename)

    # def set_gis(self, filename: str,
    #             hru_layer: Optional[str] = None,
    #             hru_key: Optional[str] = None,
    #             seg_layer: Optional[str] = None,
    #             seg_key: Optional[str] = None,
    #             ):
    #
    #     if hru_layer:
    #         if self.__verbose:
    #             print('Reading HRU polygons')
    #         self.__hru_poly = read_gis(filename, hru_layer)
    #         self.__hru_shape_key = hru_key
    #
    #     if seg_layer:
    #         if self.__verbose:
    #             print('Reading segment lines')
    #         self.__seg_poly = read_gis(filename, seg_layer)
    #         self.__seg_shape_key = seg_key
    #
    # def plot(self, name: str,
    #          output_dir: Optional[str] = None,
    #          limits: Optional[Union[str, List[float], Tuple[float, float]]] = 'valid',
    #          mask_defaults: Optional[str] = None,
    #          **kwargs):
    #     """Plot an output variable.
    #     """
    #
    #     var_data = self.get(name).iloc[0, :].to_frame(name=name)
    #
    #     if isinstance(limits, str):
    #         # if limits == 'valid':
    #         #     # Use the defined valid range of possible values
    #         #     drange = [cparam.minimum, cparam.maximum]
    #         # elif limits == 'centered':
    #         #     # Use the maximum range of the actual data values
    #         #     lim = max(abs(cparam.data.min().min()), abs(cparam.data.max().max()))
    #         #     drange = [-lim, lim]
    #         if limits == 'absolute':
    #             # Use the min and max of the data values
    #             drange = [var_data.min().min(), var_data.max().max()]
    #         else:
    #             raise ValueError('String argument for limits must be "valid", "centered", or "absolute"')
    #     elif isinstance(limits, (list, tuple)):
    #         if len(limits) != 2:
    #             raise ValueError('When a list is used for plotting limits it should have 2 values (min, max)')
    #
    #         drange = [min(limits), max(limits)]
    #     else:
    #         raise TypeError('Argument, limits, must be string or a list[min,max]')
    #
    #     cmap, norm = set_colormap(name, var_data, min_val=drange[0],
    #                               max_val=drange[1], **kwargs)
    #
    #     if self.__hru_poly is not None:
    #         # Get extent information
    #         minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds
    #
    #         crs_proj = get_projection(self.__hru_poly)
    #
    #         # Takes care of multipolygons that are in the NHM geodatabase/shapefile
    #         geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)
    #
    #         # print('Writing first plot')
    #         df_mrg = geoms_exploded.merge(var_data, left_on=self.__hru_shape_key, right_index=True, how='left')
    #
    #         fig_width, fig_height = get_figsize([minx, maxx, miny, maxy], **dict(kwargs))
    #         kwargs.pop('init_size', None)
    #
    #         fig = plt.figure(figsize=(fig_width, fig_height))
    #
    #         ax = plt.axes(projection=crs_proj)
    #
    #         # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
    #
    #         ax = plt.axes(projection=crs_proj)
    #
    #         try:
    #             ax.coastlines()
    #         except AttributeError:
    #             pass
    #
    #         gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    #         gl.top_labels = None
    #         gl.right_labels = None
    #         gl.xformatter = LONGITUDE_FORMATTER
    #         gl.yformatter = LATITUDE_FORMATTER
    #         # ax.gridlines()
    #         ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)
    #
    #         mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    #         mapper.set_array(df_mrg[name])
    #         cax = fig.add_axes((ax.get_position().x1 + 0.01,
    #                             ax.get_position().y0, 0.02,
    #                             ax.get_position().height))
    #
    #         plt.colorbar(mapper, cax=cax)   # , label=cparam.units)
    #         plt.title(f'Variable: {name}')
    #
    #         col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
    #                                       **dict(kwargs, cmap=cmap, norm=norm))
