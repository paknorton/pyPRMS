

try:
    # NOTE: cached_property is not available in python version < 3.8
    from functools import cached_property
except ImportError:
    pass

import gc
import geopandas    # type: ignore
import networkx as nx   # type: ignore
import numpy as np
import pandas as pd     # type: ignore
from collections import OrderedDict
# from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType

try:
    from typing import Optional, Union, Dict, List, Tuple, OrderedDict as OrderedDictType
except ImportError:
    # pre-python 3.7.2
    from typing import Optional, Union, Dict, List, MutableMapping as OrderedDictType   # type: ignore

import matplotlib.pyplot as plt     # type: ignore
# import matplotlib.colors as colors
import matplotlib as mpl        # type: ignore

from pyPRMS.Parameter import Parameter
from pyPRMS.plot_helpers import set_colormap, get_projection, plot_line_collection, plot_polygon_collection
from pyPRMS.Exceptions_custom import ParameterError


class Parameters(object):
    """Container of multiple pyPRMS.Parameter objects.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2017-05-01

    def __init__(self):
        """Initialize the Parameters object.

        Create an ordered dictionary to contain pyPRMS.Parameter objects
        """
        self.__parameters = OrderedDict()
        self.__hru_poly = None
        self.__hru_shape_key = None
        self.__seg_poly = None
        self.__seg_shape_key = None
        self.__seg_to_hru = None
        self.__hru_to_seg = None

    def __getattr__(self, name):
        """Not sure what to write yet.
        """

        # Undefined attributes will look up the given parameter
        # return self.get(item)
        return getattr(self.__parameters, name)

    def __getitem__(self, item):
        """Not sure what to write yet.
        """

        return self.get(item)

    try:
        @cached_property
        def hru_to_seg(self) -> OrderedDictType[int, int]:
            """Returns an ordered dictionary mapping NHM HRU IDs to HRU NHM segment IDs.

            :returns: dictionary mapping nhm_id to hru_segment_nhm
            """

            self.__hru_to_seg = OrderedDict()

            hru_segment = self.__parameters['hru_segment_nhm'].tolist()
            nhm_id = self.__parameters['nhm_id'].tolist()

            for ii, vv in enumerate(hru_segment):
                # keys are 1-based, values in arrays are 1-based
                self.__hru_to_seg[nhm_id[ii]] = vv
            return self.__hru_to_seg
    except NameError:
        # Python pre-3.8 does not have the cache_property decorator
        def hru_to_seg(self) -> OrderedDictType[int, int]:
            """Returns an ordered dictionary mapping HRU IDs to HRU segment IDs.

            :returns: dictionary mapping nhm_id to hru_segment_nhm
            """

            self.__hru_to_seg = OrderedDict()

            hru_segment = self.__parameters['hru_segment_nhm'].tolist()
            nhm_id = self.__parameters['nhm_id'].tolist()

            for ii, vv in enumerate(hru_segment):
                # keys are 1-based, values in arrays are 1-based
                self.__hru_to_seg[nhm_id[ii]] = vv
            return self.__hru_to_seg

    @property
    def parameters(self) -> OrderedDictType[str, Parameter]:
        """Returns an ordered dictionary of parameter objects.

        :returns: dictionary of Parameter objects
        """

        return self.__parameters

    @property
    def poi_to_seg(self) -> Dict[str, int]:
        """Returns a dictionary mapping poi_id to poi_seg.

        :returns: dictionary mapping poi_id to poi_seg"""

        return OrderedDict(zip(self.__parameters['poi_gage_id'].data,
                        self.__parameters['poi_gage_segment'].data))

    @property
    def poi_to_seg0(self):
        """Returns a dictionary mapping poi_id to zero-based poi_seg.

        :returns: dictionary mapping poi_id to zero-based poi_seg"""

        return OrderedDict(zip(self.__parameters['poi_gage_id'].data,
                        self.__parameters['poi_gage_segment'].data - 1))

    try:
        @cached_property
        def seg_to_hru(self) -> OrderedDictType[int, int]:
            """Returns an ordered dictionary mapping HRU segment IDs to HRU IDs.

            :returns: dictionary mapping hru_segment_nhm to nhm_id
            """

            self.__seg_to_hru = OrderedDict()

            hru_segment = self.__parameters['hru_segment_nhm'].tolist()
            nhm_id = self.__parameters['nhm_id'].tolist()

            for ii, vv in enumerate(hru_segment):
                # keys are 1-based, values in arrays are 1-based
                # Non-routed HRUs have a seg key = zero
                self.__seg_to_hru.setdefault(vv, []).append(nhm_id[ii])
            return self.__seg_to_hru
    except NameError:
        def seg_to_hru(self) -> OrderedDictType[int, int]:
            """Returns an ordered dictionary mapping HRU segment IDs to HRU IDs.

            :returns: dictionary mapping hru_segment_nhm to nhm_id
            """

            self.__seg_to_hru = OrderedDict()

            hru_segment = self.__parameters['hru_segment_nhm'].tolist()
            nhm_id = self.__parameters['nhm_id'].tolist()

            for ii, vv in enumerate(hru_segment):
                # keys are 1-based, values in arrays are 1-based
                # Non-routed HRUs have a seg key = zero
                self.__seg_to_hru.setdefault(vv, []).append(nhm_id[ii])
            return self.__seg_to_hru

    def add(self, name: Optional[str] = None,
            datatype: Optional[int] = None,
            units: Optional[str] = None,
            model: Optional[str] = None,
            description: Optional[str] = None,
            help: Optional[str] = None,
            modules: Optional[Union[str, List[str]]] = None,
            minimum: Optional[Union[int, float, str]] = None,
            maximum: Optional[Union[int, float, str]] = None,
            default: Optional[Union[int, float, str]] = None,
            info: Optional[Parameter] = None):

        """Add a new parameter by name.

        :param name: A valid PRMS parameter name
        :param datatype: The datatype for the parameter (1-Integer, 2-Float, 3-Double, 4-String)
        :param units: Option units string for the parameter
        :param model: Name of model parameter is valid for
        :param description: Description of the parameter
        :param help: Help text for the parameter
        :param modules: List of modules that require the parameter
        :param minimum: Minimum value allowed in the parameter data
        :param maximum: Maximum value allowed in the parameter data
        :param default: Default value used for parameter data
        :param info: Parameter object containing the metadata information for the parameter

        :raises ParameterError: if parameter already exists or name is None
        """

        # Add a new parameter
        if self.exists(name):
            raise ParameterError("Parameter already exists")
        elif name is None:
            raise ParameterError("None is not a valid parameter name")

        if isinstance(info, Parameter):
            self.__parameters[name] = Parameter(name=name,
                                                datatype=info.datatype,
                                                units=info.units,
                                                model=info.model,
                                                description=info.description,
                                                help=info.help,
                                                modules=info.modules,
                                                minimum=info.minimum,
                                                maximum=info.maximum,
                                                default=info.default)
        else:
            self.__parameters[name] = Parameter(name=name, datatype=datatype, units=units,
                                                model=model, description=description,
                                                help=help, modules=modules,
                                                minimum=minimum, maximum=maximum,
                                                default=default)

    def check(self):
        """Check all parameter variables for proper array size.
        """

        # for pp in self.__parameters.values():
        for pk in sorted(list(self.__parameters.keys())):
            pp = self.__parameters[pk]

            # print(pp.check())
            if pp.has_correct_size():
                print(f'{pk}: Size OK')
            else:
                print(f'{pk}: Incorrect number of values for dimensions.')

            if not pp.check_values():
                pp_stats = pp.stats()

                if not(isinstance(pp.minimum, str) or isinstance(pp.maximum, str)):
                    print(f'    WARNING: Value(s) (range: {pp_stats.min}, {pp_stats.max}) outside ' +
                          f'the valid range of ({pp.minimum}, {pp.maximum})')
                    # print(f'    WARNING: Value(s) (range: {pp.data.min()}, {pp.data.max()}) outside ' +
                    #       f'the valid range of ({pp.minimum}, {pp.maximum})')
                elif pp.minimum == 'bounded':
                    # TODO: Handling bounded parameters needs improvement
                    print(f'    WARNING: Bounded parameter value(s) (range: {pp_stats.min}, {pp_stats.max}) outside ' +
                          f'the valid range of ({pp.default}, {pp.maximum})')
                    # print(f'    WARNING: Bounded parameter value(s) (range: {pp.data.min()}, {pp.data.max()}) outside ' +
                    #       f'the valid range of ({pp.default}, {pp.maximum})')

            if pp.all_equal():
                if pp.data.ndim == 2:
                    print('    INFO: dimensioned [{1}, {2}]; all values by {1} are equal to {0}'.format(pp.data[0],
                                                                                                        *list(pp.dimensions.keys())))
                else:
                    print('    INFO: dimensioned [{1}]; all values are equal to {0}'.format(pp.data[0],
                                                                                            *list(pp.dimensions.keys())))

            if pp.name == 'snarea_curve':
                if pp.as_dataframe.values.reshape((-1, 11)).shape[0] != self.__parameters['hru_deplcrv'].unique().size:
                    print('  WARNING: snarea_curve has more entries than needed by hru_deplcrv')

    def exists(self, name) -> bool:
        """Checks if a parameter name exists.

        :param str name: Name of the parameter
        :returns: True if parameter exists, otherwise False
        """
        return name in self.parameters.keys()

    def get(self, name: str) -> Parameter:
        """Returns a parameter object.

        :param name: The name of the parameter
        :returns: Parameter object
        """

        # Return the given parameter
        if self.exists(name):
            return self.__parameters[name]
        # TODO: This shouldn't be a value error
        raise ValueError(f'Parameter, {name}, does not exist.')

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Returns a pandas DataFrame for a parameter.

        If the parameter dimensions includes either nhrus or nsegment then the
        respective national ids are included, if they exist, as the index in the
        returned dataframe.

        :param name: The name of the parameter
        :returns: Pandas DataFrame of the parameter data
        """

        if not self.exists(name):
            raise ValueError(f'Parameter, {name}, has no associated data')

        cparam = self.__parameters[name]
        param_data = cparam.as_dataframe

        if set(cparam.dimensions.keys()).intersection({'nhru', 'ngw', 'nssr'}):
            if name != 'nhm_id':
                try:
                    param_id = self.__parameters['nhm_id'].as_dataframe

                    # Create a DataFrame of the parameter
                    param_data = param_data.merge(param_id, left_index=True, right_index=True)
                    param_data.set_index('nhm_id', inplace=True)
                except KeyError:
                    # If there is no nhm_id parameter then just return the
                    # requested parameter by itself
                    param_data.rename(index={k: k + 1 for k in param_data.index},
                                      inplace=True)
                    param_data.index.name = 'hru'
            else:
                param_data = self.__parameters['nhm_id'].as_dataframe
        elif set(cparam.dimensions.keys()).intersection({'nsegment'}):
            try:
                param_id = self.__parameters['nhm_seg'].as_dataframe

                # Create a DataFrame of the parameter
                param_data = param_data.merge(param_id, left_index=True, right_index=True)
                param_data.set_index('nhm_seg', inplace=True)
            except KeyError:
                param_data.rename(index={k: k + 1 for k in param_data.index}, inplace=True)
                param_data.index.name = 'seg'
        elif name == 'snarea_curve':
            # Special handling for snarea_curve parameter
            param_data = pd.DataFrame(cparam.as_dataframe.values.reshape((-1, 11)))
            param_data.rename(columns={k: k+1 for k in param_data.columns},
                              index={k: k+1 for k in param_data.index},
                              inplace=True)
            param_data.index.name = 'curve_index'
        return param_data

    def get_subset(self, name: str, global_ids: List[int]) -> pd.DataFrame:
        """Returns a subset for a parameter based on the global_ids (e.g. nhm).

        :param name: Name of the parameter
        :param global_ids: List of global IDs to extract
        :returns: Dataframe of extracted values
        """
        param = self.__parameters[name]
        dim_set = set(param.dimensions.keys()).intersection({'nhru', 'nssr', 'ngw', 'nsegment'})
        id_index_map = {}
        cdim = dim_set.pop()

        if cdim in ['nhru', 'nssr', 'ngw']:
            # Global IDs should be in the range of nhm_id
            id_index_map = self.__parameters['nhm_id'].index_map
        elif cdim in ['nsegment']:
            # Global IDs should be in the range of nhm_seg
            id_index_map = self.__parameters['nhm_seg'].index_map

        # Zero-based indices in order of global_ids
        nhm_idx0 = []
        for kk in global_ids:
            nhm_idx0.append(id_index_map[kk])

        if param.dimensions.ndims == 2:
            return param.data[tuple(nhm_idx0), :]
        else:
            return param.data[tuple(nhm_idx0), ]

    def plot(self, name: str,
             output_dir: Optional[str] = None,
             limits: Optional[Union[str, List[float], Tuple[float, float]]] = 'valid',
             mask_defaults: Optional[str] = None,
             **kwargs):
        """Plot a parameter.

        Plots either to the screen or an output directory.

        :param name: Name of parameter to plot
        :param output_dir: Directory to write plot to (None for write to screen only)
        :param limits: Limits to use for colorbar. One of 'valid', 'centered', 'absolute', or list of floats. Default is 'valid'.
        :param mask_defaults: Color for defaults values
        """

        # is_monthly = False
        time_index = None

        if self.exists(name):
            cparam = self.__parameters[name]

            if set(cparam.dimensions.keys()).intersection({'nmonths'}):
                # Need 12 monthly plots of parameter
                # is_monthly = True
                time_index = 0  # starting time index
                param_data = self.get_dataframe(name).iloc[:, time_index].to_frame(name=name)
            else:
                param_data = self.get_dataframe(name).iloc[:]

            if mask_defaults is not None:
                param_data = param_data.mask(param_data == cparam.default)

            if isinstance(limits, str):
                if limits == 'valid':
                    # Use the defined valid range of possible values
                    if cparam.minimum == 'bounded':
                        # Parameters with bounded values need to always use the actual range of values
                        drange = [cparam.data.min().min(), cparam.data.max().max()]
                    elif name == 'jh_coef':
                        drange = [-0.05, 0.05]
                    else:
                        drange = [cparam.minimum, cparam.maximum]
                elif limits == 'centered':
                    # Use the maximum range of the actual data values
                    lim = max(abs(cparam.data.min().min()), abs(cparam.data.max().max()))
                    drange = [-lim, lim]
                elif limits == 'absolute':
                    # Use the min and max of the data values
                    drange = [cparam.data.min().min(), cparam.data.max().max()]
                else:
                    raise ValueError('String argument for limits must be "valid", "centered", or "absolute"')
            elif isinstance(limits, (list, tuple)):
                if len(limits) != 2:
                    raise ValueError('When a list is used for plotting limits it should have 2 values (min, max)')

                drange = [min(limits), max(limits)]
            else:
                raise TypeError('Argument, limits, must be string or a list[min,max]')

            cmap, norm = set_colormap(name, param_data, min_val=drange[0],
                                      max_val=drange[1], **kwargs)

            if mask_defaults is not None:
                cmap.set_bad(mask_defaults, 0.7)

            if set(cparam.dimensions.keys()).intersection({'nhru', 'ngw', 'nssr'}):
                # Get extent information
                minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds

                crs_proj = get_projection(self.__hru_poly)

                # This takes care of multipolygons that are in the NHM geodatabase/shapefile
                geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)

                # print('Writing first plot')
                df_mrg = geoms_exploded.merge(param_data, left_on=self.__hru_shape_key, right_index=True, how='left')

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

                ax = plt.axes(projection=crs_proj)
                ax.coastlines()
                ax.gridlines()
                ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

                mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                mapper.set_array(df_mrg[name])

                # TODO: 2022-06-17 PAN - Categorical variables require entry in two places.
                #       The first place is here for labelling and the second place is in
                #       plot_helpers.py for adjusting the color boundaries. Need to figure out
                #       a more consistent/better way to do this.
                if name == 'hru_deplcrv':
                    # tck_arr = np.arange(param_data.min().min(), param_data.max().max()+1)
                    tck_arr = np.arange(drange[0], drange[1]+1)
                    cb = plt.colorbar(mapper, shrink=0.6, ticks=tck_arr, label='Curve index')
                    cb.ax.tick_params(length=0)
                elif name == 'soil_type':
                    tck_arr = np.arange(drange[0], drange[1]+1)
                    cb = plt.colorbar(mapper, shrink=0.6, ticks=tck_arr, label=name)
                    cb.ax.tick_params(length=0)
                else:
                    plt.colorbar(mapper, shrink=0.6, label=cparam.units)

                # if is_monthly:
                if time_index is not None:
                    plt.title(f'Variable: {name},  Month: {time_index+1}')
                else:
                    plt.title(f'Variable: {name}')

                col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
                                              **dict(kwargs, cmap=cmap, norm=norm))

                if mask_defaults is not None:
                    plt.annotate(f'NOTE: Values = {cparam.default} are masked', xy=(0.5, 0.01),
                                 xycoords='axes fraction', fontsize=12, fontweight='bold',
                                 bbox=dict(facecolor=mask_defaults, alpha=1.0))

                    # plt.text(x, y, 'Barcelona',fontsize=12,fontweight='bold', ha='left',
                    #          va='center',color='k', bbox=dict(facecolor='b', alpha=0.2))

                if output_dir is not None:
                    # if is_monthly:
                    if time_index is not None:
                        # First month
                        plt.savefig(f'{output_dir}/{name}_{time_index+1:02}.png', dpi=150, bbox_inches='tight')

                        for tt in range(1, 12):
                            # Months 2 through 12
                            # print(f'    Index: {tt}')
                            param_data = self.get_dataframe(name).iloc[:, tt].to_frame(name=name)

                            if mask_defaults is not None:
                                param_data = param_data.mask(param_data == cparam.default)

                            df_mrg = geoms_exploded.merge(param_data, left_on=self.__hru_shape_key, right_index=True,
                                                          how='left')

                            # if is_monthly:
                            # if time_index is not None:
                            ax.set_title(f'Variable: {name},  Month: {tt+1}')

                            col.set_array(df_mrg[name])
                            # fig
                            plt.savefig(f'{output_dir}/{name}_{tt+1:02}.png', dpi=150, bbox_inches='tight')
                    else:
                        plt.savefig(f'{output_dir}/{name}.png', dpi=150, bbox_inches='tight')

                    # Close the figure so we don't chew up memory
                    fig.clf()
                    plt.close()
                    gc.collect()
            elif set(cparam.dimensions.keys()).intersection({'nsegment'}):
                # Plot segment parameters
                # Get extent information
                if self.__seg_poly is not None:
                    if self.__hru_poly is not None:
                        minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds
                        hru_geoms_exploded = self.__hru_poly.explode().reset_index(level=1, drop=True)
                    else:
                        minx, miny, maxx, maxy = self.__seg_poly.geometry.total_bounds

                    seg_geoms_exploded = self.__seg_poly.explode().reset_index(level=1, drop=True)

                    # param_data = self.get_dataframe(name).iloc[:]

                    crs_proj = get_projection(self.__seg_poly)

                    # print('Writing first plot')
                    df_mrg = seg_geoms_exploded.merge(param_data, left_on=self.__seg_shape_key,
                                                      right_index=True, how='left')

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

                    ax = plt.axes(projection=crs_proj)
                    ax.coastlines()
                    ax.gridlines()
                    ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

                    if kwargs.get('vary_color', True):
                        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                        mapper.set_array(df_mrg[name])
                        plt.colorbar(mapper, shrink=0.6, label=cparam.units)

                    plt.title('Variable: {}'.format(name))

                    # TODO: 2022-06-16 PAN - figure out best way to optionally include HRUs in
                    #       segment plots
                    # if self.__hru_poly is not None:
                    #     hru_poly = plot_polygon_collection(ax, hru_geoms_exploded.geometry, **dict(kwargs, cmap=cmap,
                    #                                                                                norm=norm,
                    #                                                                                linewidth=0.5,
                    #                                                                                alpha=0.7))

                    col = plot_line_collection(ax, df_mrg.geometry, values=df_mrg[name],
                                               **dict(kwargs, cmap=cmap, norm=norm))

                    if mask_defaults is not None:
                        plt.annotate(f'NOTE: Values = {cparam.default} are masked', xy=(0.5, 0.01),
                                     xycoords='axes fraction', fontsize=12, fontweight='bold',
                                     bbox=dict(facecolor=mask_defaults, alpha=1.0))

                    if output_dir is not None:
                        plt.savefig(f'{output_dir}/{name}.png', dpi=150, bbox_inches='tight')

                        # Close the figure so we don't chew up memory
                        fig.clf()
                        plt.close()
                        gc.collect()
                else:
                    print('No segment shapefile is loaded; skipping')
            else:
                print('Non-plottable parameter')

    def remove(self, name: Union[str, List[str]]):
        """Delete one or more parameters if they exist.

        :param name: parameter or list of parameters to remove
        """

        if isinstance(name, list):
            # Remove multiple parameters
            for cparam in name:
                if self.exists(cparam):
                    del self.__parameters[cparam]
            pass
        else:
            if self.exists(name):
                del self.__parameters[name]


    def remove_poi(self, poi: str):
        """Remove POIs by gage_id.

        :param poi: POI id to remove
        """

        # First get array of poi_gage_id indices matching the specified POI IDs
        poi_ids = self.get('poi_gage_id').data
        sorter = np.argsort(poi_ids)
        poi_del_indices = sorter[np.searchsorted(poi_ids, poi, sorter=sorter)]

        poi_parameters = ['poi_gage_id', 'poi_gage_segment', 'poi_type']

        # print(f'POIs to delete: {poi}')
        # print(f'Current POIs: {poi_ids}')
        # print(f'Size of poi_del_indices: {poi_del_indices.size}')
        if self.get('poi_gage_id').dimensions.get('npoigages').size == poi_del_indices.size:
            # We're trying to remove all the POIs
            for pp in poi_parameters:
                self.remove(pp)
        else:
            # Remove the matching poi gage entries from each of the poi-related parameters
            for pp in poi_parameters:
                self.get(pp).remove_by_index('npoigages', poi_del_indices)


    def remove_by_global_id(self, hrus: Optional[List] = None,
                            segs: Optional[List] = None):
        """Removes data-by-id (nhm_seg, nhm_id) from all parameters.

        :param hrus: list of national HRU ids
        :param segs: list of national segment ids
        """

        if segs is not None:
            pass

        if hrus is not None:
            # Map original nhm_id to their index
            nhm_idx = OrderedDict((hid, ii) for ii, hid in enumerate(self.get('nhm_id').data.tolist()))
            nhm_seg = self.get('nhm_seg').tolist()

            print(list(nhm_idx.keys())[0:10])

            for xx in list(nhm_idx.keys()):
                if xx in hrus:
                    del nhm_idx[xx]

            print('-'*40)
            print(list(nhm_idx.keys())[0:10])
            print(list(nhm_idx.values())[0:10])

            # [hru_segment_nhm[yy] for yy in nhm_idx.values()]
            self.get('nhm_id').subset_by_index('nhru', nhm_idx.values())

            # Update hru_segment_nhm then go back and make sure the referenced nhm_segs are valid
            self.get('hru_segment_nhm').subset_by_index('nhru', nhm_idx.values())
            self.get('hru_segment_nhm').data = [kk if kk in nhm_seg else 0 if kk == 0 else -1
                                                for kk in self.get('hru_segment_nhm').data.tolist()]

            # Now do the local hru_segment
            self.get('hru_segment').subset_by_index('nhru', nhm_idx.values())
            self.get('hru_segment').data = [nhm_seg.index(kk)+1 if kk in nhm_seg else 0 if kk == 0 else -1
                                            for kk in self.get('hru_segment_nhm').data.tolist()]

            # # First remove the HRUs from nhm_id and hru_segment_nhm
            # id_to_seg = np.column_stack((self.get('nhm_id').data, self.get('hru_segment_nhm').data))
            #
            # # Create ordered dictionary to reindex hru_segment
            # nhm_id_to_hru_segment_nhm = OrderedDict((nhm, hseg) for nhm, hseg in id_to_seg)
            #
            # nhm_seg = self.get('nhm_seg').data.tolist()
            #
            # self.get('nhm_id').data = [xx for xx in nhm_id_to_hru_segment_nhm.keys()]
            # # self.get('nhm_id').remove_by_index('nhru', hrus)
            #
            # self.get('hru_segment_nhm').data = [kk if kk in nhm_seg else 0 if kk == 0 else -1
            #                                     for kk in nhm_id_to_hru_segment_nhm.values()]
            #
            # self.get('hru_segment').data = [nhm_seg.index(kk)+1 if kk in nhm_seg else 0 if kk == 0 else -1
            #                                 for kk in nhm_id_to_hru_segment_nhm.values()]

            for pp in self.__parameters.values():
                if pp.name not in ['nhm_id', 'hru_segment_nhm', 'hru_segment']:
                    dim_set = set(pp.dimensions.keys()).intersection({'nhru', 'nssr', 'ngw'})

                    if bool(dim_set):
                        if len(dim_set) > 1:
                            raise ValueError('dim_set > 1 for {}'.format(pp.name))
                        else:
                            cdim = dim_set.pop()
                            pp.subset_by_index(cdim, nhm_idx.values())

                            if pp.name == 'hru_deplcrv':
                                # Save the list of snow indices for reducing the snarea_curve later
                                uniq_deplcrv_idx = list(set(pp.data.tolist()))
                                uniq_dict = {}
                                for ii, xx in enumerate(uniq_deplcrv_idx):
                                    uniq_dict[xx] = ii + 1

                                uniq_deplcrv_idx0 = [xx - 1 for xx in uniq_deplcrv_idx]

                                # Renumber the hru_deplcrv indices
                                data_copy = pp.data.copy()
                                with np.nditer(data_copy, op_flags=['readwrite']) as it:
                                    for xx in it:
                                        xx[...] = uniq_dict[int(xx)]

                                pp.data = data_copy

                                tmp = self.__parameters['snarea_curve'].data.reshape((-1, 11))[tuple(uniq_deplcrv_idx0), :]

                                self.__parameters['snarea_curve'].data = tmp.ravel()
                                self.__parameters['snarea_curve'].dimensions['ndeplval'].size = tmp.size

            # Need to reduce the snarea_curve array to match the number of indices in hru_deplcrv
            # new_deplcrv = pp['hru_deplcrv'].data.tolist()

    def shapefile_segments(self, filename: str,
                           layer_name: Optional[str] = None,
                           shape_key: Optional[str] = None):
        """Read a shapefile or geodatabase that corresponds to stream segments.

        :param filename: name of shapefile or geodatabase
        :param layer_name: name of layer in geodatabase
        :param shape_key: name of attribute for key
        """

        self.__seg_poly = geopandas.read_file(filename, layer=layer_name)

        if self.__seg_poly.crs.name == 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version':
            print('Overriding USGS aea crs with EPSG:5070')
            self.__seg_poly.crs = 'EPSG:5070'
        self.__seg_shape_key = shape_key

    def shapefile_hrus(self, filename: str,
                       layer_name: Optional[str] = None,
                       shape_key: Optional[str] = None):
        """Read a shapefile or geodatabase that corresponds to HRUs.

        :param filename: name of shapefile or geodatabase
        :param layer_name: name of layer in geodatabase
        :param shape_key: name of attribute for key
        """

        self.__hru_poly = geopandas.read_file(filename, layer=layer_name)

        if self.__hru_poly.crs.name == 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version':
            print('Overriding USGS aea crs with EPSG:5070')
            self.__hru_poly.crs = 'EPSG:5070'
        self.__hru_shape_key = shape_key

    def stream_network(self, tosegment: Optional[str] = 'tosegment_nhm',
                       seg_id: Optional[str] = 'nhm_seg') -> Union[nx.DiGraph, None]:
        """
        Create Directed, Acyclic Graph (DAG) of stream network.

        :param tosegment: name of parameter to use for HRU tosegment
        :param seg_id: name of parameter to use for the segment IDs
        :returns: directed-acyclic-graph (DAG)
        """

        if self.exists(tosegment) and self.exists(seg_id):
            seg = self.__parameters.get(seg_id).tolist()
            toseg = self.__parameters.get(tosegment).tolist()

            dag_ds = nx.DiGraph()
            for ii, vv in enumerate(toseg):
                #     dag_ds.add_edge(ii+1, vv)
                if vv == 0:
                    dag_ds.add_edge(seg[ii], 'Out_{}'.format(seg[ii]))
                else:
                    dag_ds.add_edge(seg[ii], vv)

            return dag_ds
        return None

    def update_element(self, name: str,
                       id1: int,
                       value: Union[int, float, List[int], List[float]]):
        """Update single value or row of values (e.g. nhru by nmonths) for a
        given nhm_id or nhm_seg.

        :param name: name of parameter to update
        :param id1: scalar nhm_id or nhm_seg
        :param value: new value(s)
        """

        # NOTE: id1 is either an nhm_id or nhm_seg (both are 1-based)
        cparam = self.get(name)

        if cparam.is_hru_param():
            # Lookup index for nhm_id
            idx0 = self.get('nhm_id')._value_index(id1)

            if idx0 is None:
                raise TypeError(f'nhm_id _value_index returned NoneType instead of ndarray')

            if len(idx0) > 1:
                raise ValueError(f'nhm_id values should be unique')
            else:
                cparam.update_element(idx0[0], value)
        elif cparam.is_seg_param():
            # Lookup index for nhm_seg
            idx0 = self.get('nhm_seg')._value_index(id1)

            if idx0 is None:
                raise TypeError(f'nhm_seg _value_index returned NoneType instead of ndarray')

            if len(idx0) > 1:
                raise ValueError(f'nhm_seg values should be unique')
            else:
                cparam.update_element(idx0[0], value)

        # TODO: Add handling for other dimensions

    @staticmethod
    def _get_upstream_subset(dag_ds: nx.DiGraph,
                             cutoff_segs: List[int],
                             outlet_segs: List[int]):
        """Create upstream stream network bounded by downstream segment(s) and upstream cutoff(s).

        :param dag_ds: stream network
        :param cutoff_segs: list of nhm_seg IDs above which the network is truncated
        :param outlet_segs: list of nhm_seg IDs for the downstream-most segments
        """

        # Create the upstream graph.
        dag_us = dag_ds.reverse()
        # bandit_helper_log.debug('Number of NHM upstream nodes: {}'.format(dag_us.number_of_nodes()))
        # bandit_helper_log.debug('Number of NHM upstream edges: {}'.format(dag_us.number_of_edges()))

        # Trim the u/s graph to remove segments above the u/s cutoff segments
        try:
            for xx in cutoff_segs:
                try:
                    dag_us.remove_nodes_from(nx.dfs_predecessors(dag_us, xx))

                    # Also remove the cutoff segment itself
                    dag_us.remove_node(xx)
                except KeyError:
                    print('WARNING: nhm_segment {} does not exist in stream network'.format(xx))
        except TypeError:
            # bandit_helper_log.error('\nSelected cutoffs should at least be an empty list instead of NoneType.')
            exit(200)

        # bandit_helper_log.debug('Number of NHM upstream nodes (trimmed): {}'.format(dag_us.number_of_nodes()))
        # bandit_helper_log.debug('Number of NHM upstream edges (trimmed): {}'.format(dag_us.number_of_edges()))

        # =======================================
        # Given a d/s segment (dsmost_seg) create a subset of u/s segments

        # Get all unique segments u/s of the starting segment
        uniq_seg_us = set()
        if outlet_segs:
            for xx in outlet_segs:
                try:
                    pred = nx.dfs_predecessors(dag_us, xx)
                    uniq_seg_us = uniq_seg_us.union(set(pred.keys()).union(set(pred.values())))
                except KeyError:
                    # bandit_helper_log.error('KeyError: Segment {} does not exist in stream network'.format(xx))
                    print(f'\nKeyError: Segment {xx} does not exist in stream network')

            # Get a subgraph in the dag_ds graph and return the edges
            dag_ds_subset = dag_ds.subgraph(uniq_seg_us).copy()

            # 2018-02-13 PAN: It is possible to have outlets specified which are not truly
            #                 outlets in the most conservative sense (e.g. a point where
            #                 the stream network exits the study area). This occurs when
            #                 doing headwater extractions where all segments for a headwater
            #                 are specified in the configuration file. Instead of creating
            #                 output edges for all specified 'outlets' the set difference
            #                 between the specified outlets and nodes in the graph subset
            #                 which have no edges is performed first to reduce the number of
            #                 outlets to the 'true' outlets of the system.
            node_outlets = [ee[0] for ee in dag_ds_subset.edges()]
            true_outlets = set(outlet_segs).difference(set(node_outlets))
            # bandit_helper_log.debug('node_outlets: {}'.format(','.join(map(str, node_outlets))))
            # bandit_helper_log.debug('true_outlets: {}'.format(','.join(map(str, true_outlets))))

            # Add the downstream segments that exit the subgraph
            for xx in true_outlets:
                nhm_outlet = list(dag_ds.neighbors(xx))[0]
                dag_ds_subset.add_node(nhm_outlet, style='filled', fontcolor='white', fillcolor='grey')
                dag_ds_subset.add_edge(xx, nhm_outlet)
                dag_ds_subset.nodes[xx]['style'] = 'filled'
                dag_ds_subset.nodes[xx]['fontcolor'] = 'white'
                dag_ds_subset.nodes[xx]['fillcolor'] = 'blue'
        else:
            # No outlets specified so pull the CONUS
            dag_ds_subset = dag_ds

        return dag_ds_subset

    # def replace_values(self, varname, newvals, newdims=None):
    #     """Replaces all values for a given variable/parameter. Size of old and new arrays/values must match."""
    #     if not self.__isloaded:
    #         self.load_file()
    #
    #     # parent = self.__paramdict['Parameters']
    #     thevar = self.get_var(varname)
    #
    #     # NOTE: Need to figure out whether this function should expect row-major ordering
    #     #       or column-major ordering when called. Right it expects column-major ordering
    #     #       for newvals, which means no re-ordering of the array is necessary when
    #     #       replacing values.
    #     if newdims is None:
    #         # We are not changing dimensions of the variable/parameter, just the values
    #         # Check if size of newvals array matches the oldvals array
    #         if isinstance(newvals, list) and len(newvals) == thevar['values'].size:
    #             # Size of arrays match so replace the oldvals with the newvals
    #             # Lookup dimension size for each dimension name
    #             arr_shp = [self.__paramdict['Dimensions'][dd] for dd in thevar['dimnames']]
    #
    #             thevar['values'][:] = np.array(newvals).reshape(arr_shp)
    #         elif isinstance(newvals, np.ndarray) and newvals.size == thevar['values'].size:
    #             # newvals is a numpy ndarray
    #             # Size of arrays match so replace the oldvals with the newvals
    #             # Lookup dimension size for each dimension name
    #             arr_shp = [self.  __paramdict['Dimensions'][dd] for dd in thevar['dimnames']]
    #
    #             thevar['values'][:] = newvals.reshape(arr_shp)
    #         # NOTE: removed the following because even scalars should be stored as numpy array
    #         # elif thevar['values'].size == 1:
    #         #     # This is a scalar value
    #         #     if isinstance(newvals, float):
    #         #         thevar['values'] = [newvals]
    #         #     elif isinstance(newvals, int):
    #         #         thevar['values'] = [newvals]
    #         else:
    #             print("ERROR: Size of oldval array and size of newval array don't match")
    #     else:
    #         # The dimensions are being changed and new values provided
    #
    #         # Use the dimension sizes from the parameter file to check the size
    #         # of the newvals array. If the size of the newvals array doesn't match the
    #         # parameter file's dimensions sizes we have a problem.
    #         size_check = 1
    #         for dd in newdims:
    #             size_check *= self.get_dim(dd)
    #
    #         if isinstance(newvals, list) and len(newvals) == size_check:
    #             # Size of arrays match so replace the oldvals with the newvals
    #             thevar['values'] = newvals
    #             thevar['dimnames'] = newdims
    #         elif isinstance(newvals, np.ndarray) and newvals.size == size_check:
    #             # newvals is a numpy ndarray
    #             # Size of arrays match so replace the oldvals with the newvals
    #             thevar['values'] = newvals
    #             thevar['dimnames'] = newdims
    #         elif thevar['values'].size == 1:
    #             # This is a scalar value
    #             thevar['dimnames'] = newdims
    #             if isinstance(newvals, float):
    #                 thevar['values'] = [newvals]
    #             elif isinstance(newvals, int):
    #                 thevar['values'] = [newvals]
    #         else:
    #             print("ERROR: Size of newval array doesn't match dimensions in parameter file")
    #
    # def resize_dim(self, dimname, newsize):
    #     """Changes the size of the given dimension.
    #        This does *not* check validity of parameters that use the dimension.
    #        Check variable integrity before writing parameter file."""
    #
    #     # Some dimensions are related to each other.
    #     related_dims = {'ndepl': 'ndeplval', 'nhru': ['nssr', 'ngw'],
    #                     'nssr': ['nhru', 'ngw'], 'ngw': ['nhru', 'nssr']}
    #
    #     if not self.__isloaded:
    #         self.load_file()
    #
    #     parent = self.__paramdict['Dimensions']
    #
    #     if dimname in parent:
    #         parent[dimname] = newsize
    #
    #         # Also update related dimensions
    #         if dimname in related_dims:
    #             if dimname == 'ndepl':
    #                 parent[related_dims[dimname]] = parent[dimname] * 11
    #             elif dimname in ['nhru', 'nssr', 'ngw']:
    #                 for dd in related_dims[dimname]:
    #                     parent[dd] = parent[dimname]
    #         return True
    #     else:
    #         return False
    #
    # def update_values_by_hru(self, varname, newvals, hru_index):
    #     """Updates parameter/variable with new values for a a given HRU.
    #        This is used when merging data from an individual HRU into a region"""
    #     if not self.__isloaded:
    #         self.load_file()
    #
    #     # parent = self.__paramdict['Parameters']
    #     thevar = self.get_var(varname)
    #
    #     if len(newvals) == 1:
    #         thevar['values'][(hru_index - 1)] = newvals
    #     elif len(newvals) == 2:
    #         thevar['values'][(hru_index - 1), :] = newvals
    #     elif len(newvals) == 3:
    #         thevar['values'][(hru_index - 1), :, :] = newvals

# ***** END of class parameters()
