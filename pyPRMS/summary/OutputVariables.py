
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas   # type: ignore

import io
import pandas as pd   # type: ignore
import pkgutil
import xml.etree.ElementTree as xmlET

from functools import cached_property
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt     # type: ignore
import matplotlib as mpl        # type: ignore

from ..plot_helpers import set_colormap, get_projection, \
    plot_polygon_collection, read_gis

# - pass control object
# - list available variables
#     - basin (basinON_OFF = 1)
#     - streamflow (csvON_OFF = 2)
#     - segment (nsegmentON_OFF = 1,2)
#     - hru (nhruON_OFF = 1,2)
#     - <others>
# - get variable


class OutputVariables(object):
    def __init__(self,
                 control,
                 model_dir=None,
                 verbose: Optional[bool] = False,
                 ):
        """Initialize the model output object.
        """

        self.__control = control
        self.__model_dir = model_dir

        # Read the output variables metadata
        xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/variables.xml').decode('utf-8'))
        xml_tree = xmlET.parse(xml_fh)
        self.__xml_root = xml_tree.getroot()

        self.__data = None
        self.__hru_poly = None
        self.__hru_shape_key = None
        self.__seg_poly = None
        self.__seg_shape_key = None
        self.__verbose = verbose

    @cached_property
    def available_vars(self):
        """Returns dictionary of available variables and file paths"""
        var_dict = {}

        if self.__control.get('basinOutON_OFF').values == 1:
            filename = self.__control.get('basinOutBaseFileName').values
            varlist = self.__control.get('basinOutVar_names').values

            for vv in varlist:
                var_dict[vv] = f'{filename}.csv'

        if self.__control.get('nhruOutON_OFF').values in [1, 2]:
            prefix = self.__control.get('nhruOutBaseFileName').values
            varlist = self.__control.get('nhruOutVar_names').values

            for vv in varlist:
                var_dict[vv] = f'{prefix}{vv}.csv'

        if self.__control.get('nsegmentOutON_OFF').values in [1, 2]:
            prefix = self.__control.get('nsegmentOutBaseFileName').values
            varlist = self.__control.get('nsegmentOutVar_names').values

            for vv in varlist:
                var_dict[vv] = f'{prefix}{vv}.csv'

        if self.__control.get('csvON_OFF').values == 2:
            filename = self.__control.get('csv_output_file').values
            var_dict['model_streamflow'] = f'{filename}'

        return var_dict

    def get(self, varname: str) -> pd.DataFrame:
        """Get the data subset for a given variable.

        :param varname: Name of model output variable
        :returns: Model output dataframe
        """
        # df1 = pd.read_csv(byHRU_file, sep=',', header=0, index_col=0, parse_dates=True)
        # df1.columns = df1.columns.astype(int)

        if self.__model_dir:
            filename = f'{self.__model_dir}/{self.available_vars[varname]}'
        else:
            filename = f'{self.available_vars[varname]}'

        if varname == 'model_streamflow':
            poi_flds, poi_seg_flds = self._read_streamflow_header(filename)
            df = self._read_streamflow_ascii(filename, field_names=poi_flds)
        else:
            df = pd.read_csv(filename, sep=',', header=0, index_col=0, parse_dates=True)

        if varname[0:5] != 'basin' and varname != 'model_streamflow':
            df.columns = df.columns.astype(int)

        return df

    def metadata(self, varname):
        """Returns metadata dictionary for variable"""

        dtype = None
        dim = None
        desc = None
        units = None
        dimname = dim

        for elem in self.__xml_root.findall('variable'):
            # print(elem.attrib.get('name'))
            name = elem.attrib.get('name')

            if varname == name:
                dtype = elem.find('type').text

                # Convert dtype to netcdf datatype
                # dtype = NETCDF_DATATYPES[NHM_DATATYPES[dtype]]

                dim = []

                for cdim in elem.findall('.dimensions/dimension'):
                    dim = cdim.attrib.get('name')

                desc = elem.find('desc').text
                units = elem.find('units').text
                # dimname = dim_name_nc[dim]
                dimname = dim
                # dimname = dim[1:]

                break

        return {'name': varname, 'description': desc, 'units': units, 'dimension': dimname,
                'datatype': dtype}

    @staticmethod
    def _read_streamflow_header(filename):
        """Read the headers from a PRMS CSV model output file (ON_OFF=2)"""
        fhdl = open(filename, 'r')

        # First and second rows are headers
        hdr1 = fhdl.readline().strip()

        fhdl.close()

        tmp_flds = hdr1.split(' ')
        tmp_flds.remove('Date')

        flds = {nn+3: hh for nn, hh in enumerate(tmp_flds)}

        # poi_flds maps column index to POI and is used to rename the dataframe columns from indices to station IDs
        poi_flds = dict()

        # poi_seg_flds maps POI to the related segment ID
        poi_seg_flds = dict()

        for xx, yy in flds.items():
            tfld = yy.split('_')
            segid = int(tfld[2]) - 1  # Change to zero-based indices
            poiid = tfld[4]

            poi_flds[xx] = poiid
            poi_seg_flds[poiid] = segid

        return poi_flds, poi_seg_flds

    @staticmethod
    def _read_streamflow_ascii(filename, field_names):
        """Read the simulated streamflow from a PRMS CSV model output file"""
        df = pd.read_csv(filename, sep='\s+', header=None, skiprows=2, parse_dates={'time': [0, 1, 2]},
                         index_col='time')

        df.rename(columns=field_names, inplace=True)

        return df

    def set_gis(self, filename: str,
                hru_layer: Optional[str] = None,
                hru_key: Optional[str] = None,
                seg_layer: Optional[str] = None,
                seg_key: Optional[str] = None,
                ):

        if hru_layer:
            if self.__verbose:
                print('Reading HRU polygons')
            self.__hru_poly = read_gis(filename, hru_layer)
            self.__hru_shape_key = hru_key

        if seg_layer:
            if self.__verbose:
                print('Reading segment lines')
            self.__seg_poly = read_gis(filename, seg_layer)
            self.__seg_shape_key = seg_key

    def plot(self, name: str,
             output_dir: Optional[str] = None,
             limits: Optional[Union[str, List[float], Tuple[float, float]]] = 'valid',
             mask_defaults: Optional[str] = None,
             **kwargs):
        """Plot an output variable.
        """

        var_data = self.get_var(name).iloc[0, :].to_frame(name=name)

        if isinstance(limits, str):
            # if limits == 'valid':
            #     # Use the defined valid range of possible values
            #     drange = [cparam.minimum, cparam.maximum]
            # elif limits == 'centered':
            #     # Use the maximum range of the actual data values
            #     lim = max(abs(cparam.data.min().min()), abs(cparam.data.max().max()))
            #     drange = [-lim, lim]
            if limits == 'absolute':
                # Use the min and max of the data values
                drange = [var_data.min().min(), var_data.max().max()]
            else:
                raise ValueError('String argument for limits must be "valid", "centered", or "absolute"')
        elif isinstance(limits, (list, tuple)):
            if len(limits) != 2:
                raise ValueError('When a list is used for plotting limits it should have 2 values (min, max)')

            drange = [min(limits), max(limits)]
        else:
            raise TypeError('Argument, limits, must be string or a list[min,max]')

        cmap, norm = set_colormap(name, var_data, min_val=drange[0],
                                  max_val=drange[1], **kwargs)

        # Get extent information
        minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds

        crs_proj = get_projection(self.__hru_poly)

        # Takes care of multipolygons that are in the NHM geodatabase/shapefile
        geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)

        # print('Writing first plot')
        df_mrg = geoms_exploded.merge(var_data, left_on=self.__hru_shape_key, right_index=True, how='left')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

        ax = plt.axes(projection=crs_proj)
        ax.coastlines()
        ax.gridlines()
        ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mapper.set_array(df_mrg[name])

        plt.colorbar(mapper, shrink=0.6)   # , label=cparam.units)
        plt.title(f'Variable: {name}')

        col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
                                      **dict(kwargs, cmap=cmap, norm=norm))

    # @staticmethod
    # def nearest(items, pivot):
    #     return min(items, key=lambda x: abs(x - pivot))
    #
    # def read_netcdf(self):
    #     """Read model output file stored in netCDF format."""
    #
    #     self.__data = xr.open_dataset(self.__filename, chunks={})
    #
    #     try:
    #         self.__data = self.__data.assign_coords(nhru=self.__data.nhm_id)
    #     except AttributeError:
    #         pass
    #
    #     try:
    #         self.__data = self.__data.assign_coords(nsegment=self.__data.nhm_seg)
    #     except AttributeError:
    #         pass

    # def __init__(self, src_path=None, prefix=None):
    #     self.__src_path = src_path
    #     self.__prefix = prefix
    #
    # def rawcount(self, filename):
    #     f = open(filename, 'rb', buffering=0)
    #     lines = 0
    #     buf_size = 1024 * 1024
    #     read_f = f.read
    #     # read_f = f.raw.read
    #
    #     buf = read_f(buf_size)
    #     while buf:
    #         lines += buf.count(b'\n')
    #         buf = read_f(buf_size)
    #
    #     return lines
    #
    # def rawincount(self, filename):
    #     f = open(filename, 'rb', buffering=0)
    #     # bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    #     bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
    #     return sum(buf.count(b'\n') for buf in bufgen)
