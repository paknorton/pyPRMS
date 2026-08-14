from __future__ import annotations

import cartopy.crs as ccrs  # type: ignore
import gc
import matplotlib as mpl        # type: ignore
import matplotlib.pyplot as plt     # type: ignore
import netCDF4 as nc    # type: ignore
import networkx as nx   # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd     # type: ignore
import sys
import warnings
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from copy import deepcopy
from collections import defaultdict
from collections.abc import KeysView, Sequence
from functools import cached_property
from packaging.version import Version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # type: ignore

from ..control.Control import Control
from ..dimensions.Dimensions import Dimensions
from ..Exceptions_custom import ParameterError, ParameterExistsError, ParameterNotValidError
from .Parameter import Parameter, ParamDataRawType
from ..plot_helpers import set_colormap, get_projection, plot_line_collection, plot_polygon_collection, get_figsize
from ..prms_helpers import cond_check, flex_type, get_streamnet_subset
from ..constants import (CATEGORY_DELIM, DIMENSIONS_XML, external_module_map, HRU_DIMS, MetaDataType, NETCDF_DATATYPES,
                         NEW_PTYPE_TO_DTYPE, PRMS_VERSION, PRMS6_DEV_VERSION, PTYPE_TO_PRMS_TYPE, NHM_DATATYPES,
                         PARAMETERS_XML, VAR_DELIM)

from rich.table import Table

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas    # type: ignore

from ..base.console import get_console_instance

con = None

LimitOptions = Literal['valid', 'centered', 'absolute']


class Parameters(object):
    """Container of multiple pyPRMS.Parameter objects.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2017-05-01

    def __init__(self, metadata: MetaDataType,
                 verbose: bool = False):
        """Initialize the Parameters object.

        Create an ordered dictionary to contain pyPRMS.Parameter objects

        :param metadata: Metadata for the parameters
        :param verbose: Output debugging information
        """

        global con
        con = get_console_instance()
        # con.print('Parameters: Console info: {}'.format(con))

        # A full, separate copy of the original metadata dictionary
        self.__full_metadata = deepcopy(metadata)

        self.__dimensions = deepcopy(Dimensions(metadata=metadata, verbose=verbose))
        self.__parameters: dict[str, Parameter] = dict()

        self.verbose = verbose
        self.__control: Control | None = None
        self.__hru_poly = None
        self.__hru_shape_key: str | None = None
        self.__seg_poly = None
        self.__seg_shape_key: str | None = None
        self.__seg_to_hru: dict[int, list[int]] = dict()
        self.__hru_to_seg: dict[int, int] = dict()
        self.metadata = metadata['parameters']
        self.prms_version = Version(metadata['info']['version'])

    def __contains__(self, name: str) -> bool:
        """Check if a parameter exists.

        :param name: Name of the parameter
        :returns: True if parameter exists, otherwise False
        """
        return name in self.__parameters

    def __getitem__(self, item):
        """Get a parameter by name.

        :param item: Name of the parameter
        :returns: Parameter object
        """

        return self.get(item)

    def __iter__(self):
        """Iterate over parameter names.

        :returns: Iterator over parameter names
        """
        return iter(self.__parameters)

    def __len__(self) -> int:
        """Return number of parameters.

        :returns: Number of parameters
        """
        return len(self.__parameters)

    def __str__(self) -> str:
        """Pretty-print string representation of the Parameters object.

        :return: Pretty-print string of Parameters
        """

        outstr = '----- Dimensions -----\n'
        for vv in self.__dimensions.values():
            outstr += f'{vv.name}: size={vv.size}\n'

        outstr += '----- Parameters -----\n'
        for vv in self.__parameters.values():
            outstr += f'{vv.name} [{", ".join(vv.meta["dimensions"])}]\n'

        return outstr

    @property
    def control(self) -> Control | None:
        """Get Control object.

        :returns: Control object
        """

        return self.__control

    @control.setter
    def control(self, ctl_obj: Control):
        """Sets the Control object for the ParameterSet.

        :param ctl_obj: Control object
        """

        self.__control = ctl_obj

    @property
    def user_defined_units(self) -> dict[str, str]:
        """Returns a dictionary of the current unit strings for parameters
        elev_units, precip_units, runoff_units, and temp_units.

        There are four parameters (elev_units, precip_units, runoff_units, and temp_units)
        that define units used by other parameters. These unit values are
        defined as 0 or 1 which represent specific units for these parameters.
        This function returns a dictionary of these parameters with the
        string representation of the selected units (e.g. `in` or `mm` for precip_units)
        instead of the integer values.

        :returns: dictionary of the current unit strings for parameters
        """

        # Build dictionary of the *_units values.
        selected_units = {}
        for pp in ('elev_units', 'precip_units', 'runoff_units', 'temp_units'):
            try:
                cparam = self.get(pp)
                selected_units[pp] = cparam.meta['valid_values'][str(cparam.data)]
            except ParameterError:
                con.print(f'[dark_orange]WARNING[/]: {pp} parameter not found')

        return selected_units

    @property
    def dimensions(self) -> Dimensions:
        """Get dimensions object.

        :returns: Dimensions object
        """

        return self.__dimensions

    @cached_property
    def hru_to_seg(self) -> dict[int, int]:
        """Returns an ordered dictionary mapping NHM HRU IDs to HRU NHM segment IDs.

        :returns: dictionary mapping nhm_id to hru_segment_nhm
        """

        # Only supported with python >= 3.9
        hru_segment = self.get('hru_segment_nhm').data_raw
        nhm_id = self.get('nhm_id').data_raw

        self.__hru_to_seg = dict(np.rec.fromarrays([nhm_id, hru_segment]).tolist())   # type: ignore
        return self.__hru_to_seg

    @property
    def missing_params(self) -> set:
        """Get set of parameters that are required for the modules selected
        in the control file but are missing in the current set of parameters.
        """

        # TODO: 20240726 PAN - raise warning/error when certain parameters
        #                      like hru_segment_nhm, etc are missing
        pset = self._required_parameters()
        return pset.difference(set(self.parameters.keys()))

    @property
    def parameters(self) -> MappingProxyType[str, Parameter]:
        """Returns a read-only view of parameter objects.

        :returns: Read-only mapping of parameter names to Parameter objects
        """

        return MappingProxyType(self.__parameters)

    @property
    def poi_to_seg(self) -> dict[str, int]:
        """Returns a dictionary mapping poi_id to local poi_seg.

        :returns: dictionary mapping poi_id to local poi_seg"""

        gage_ids = self.get('poi_gage_id').data_raw
        gage_segments = self.get('poi_gage_segment').data_raw
        return dict(np.rec.fromarrays([gage_ids, gage_segments]).tolist())   # type: ignore

    @property
    def poi_to_seg0(self) -> dict[str, int]:
        """Returns a dictionary mapping poi_id to local, zero-based poi_seg.

        :returns: dictionary mapping poi_id to local, zero-based poi_seg"""
        gage_ids = self.get('poi_gage_id').data_raw   # .tolist()
        gage_segments = np.subtract(self.get('poi_gage_segment').data_raw, 1)
        return dict(np.rec.fromarrays([gage_ids, gage_segments]).tolist())   # type: ignore

    @cached_property
    def seg_to_hru(self) -> dict[int, list[int]]:
        """Returns a dictionary mapping HRU global segment IDs to global HRU IDs.

        Segment keys equal to zero are for non-routed HRUs. Segment keys greater than
        zero are global segment IDs.

        :returns: dictionary mapping hru_segment_nhm to nhm_id
        """

        hru_segment = self.get('hru_segment_nhm').data_raw.tolist()
        nhm_id = self.get('nhm_id').data_raw
        assert type(nhm_id) is np.ndarray and nhm_id.dtype == np.int32

        for chru_idx, cseg in enumerate(hru_segment):   # type: ignore
            # keys are 1-based, values in arrays are 1-based
            # Non-routed HRUs have a seg key = zero
            self.__seg_to_hru.setdefault(cseg, []).append(nhm_id[chru_idx].item())   # type: ignore
        return self.__seg_to_hru

    @property
    def unneeded_parameters(self) -> set:
        """Get set of parameters that are defined but not needed by any of the
        modules selected in the control file.

        :returns: set of unneeded parameter names
        """

        if self.verbose:   # pragma: no cover
            con.print('-'*20, 'unneeded_parameters', '-'*20)

        pset = self._required_parameters()
        return set(self.parameters.keys()).difference(pset)

    @property
    def xml_global_dimensions(self) -> xmlET.Element:
        """Get XML element tree of the dimensions used by all parameters.

        :returns: element tree of dimensions
        """

        dims_xml = xmlET.Element('dimensions')

        for cdim in self.dimensions.dimensions.values():
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', cdim.name)

            for kk, vv in cdim.meta.items():
                if kk in ['is_fixed', 'requires_control']:
                    pass
                else:
                    xmlET.SubElement(dim_sub, kk).text = flex_type(vv)
        return dims_xml

    @property
    def xml_global_parameters(self) -> xmlET.Element:
        """Get XML element tree of the parameters.

        :returns: element tree of parameters
        """

        # Map datatypes (e.g. np.int32) to the parameter type strings (e.g. 'I')
        inv_nhm_datatypes = {vv: kk for kk, vv in NHM_DATATYPES.items()}
        datatype_to_prms_type = {kk: inv_nhm_datatypes[vv] for kk, vv in PTYPE_TO_PRMS_TYPE.items()}

        params_xml = xmlET.Element('parameters')

        for pk in sorted(list(self.parameters.keys())):
            vv = self.get(pk)

            param_sub = xmlET.SubElement(params_xml, 'parameter')
            param_sub.set('name', vv.name)

            xmlET.SubElement(param_sub, 'type').text = datatype_to_prms_type[vv.meta['datatype']]

            for mm, md in vv.meta.items():
                if mm == 'modules':
                    modules_sub = xmlET.SubElement(param_sub, 'modules')

                    for cmod in md:
                        xmlET.SubElement(modules_sub, 'module').text = cmod
                elif mm in ['datatype', 'requires_control', 'requires_dimension', 'version']:
                    pass
                elif mm == 'dimensions':
                    param_sub.append(vv.dimensions.xml)
                else:
                    xmlET.SubElement(param_sub, mm).text = flex_type(md)

        return params_xml

    # =========================================================================
    # Methods
    def add(self, name: str):
        """Add a new parameter by name.

        :param name: A valid PRMS parameter name

        :raises ParameterError: If the parameter already exists or name is None
        """

        # Add a new parameter
        if self.exists(name):
            raise ParameterExistsError(f'{name}: Parameter already exists')

        if name not in self.metadata:
            raise ParameterNotValidError(f'{name}: Parameter is not a valid PRMS parameter')

        for cdim in self.metadata[name]['dimensions']:
            if not self.__dimensions.exists(cdim):
                raise KeyError(f'Global dimension, {cdim}, does not exist')

        if self.metadata[name].get('is_bounded', False):
            # Add the upper-bound dimension to the global dimensions
            bounded_dim_name = self.metadata[name]['maximum']
            if not self.__dimensions.exists(bounded_dim_name):
                if bounded_dim_name == 'ndepl':
                    # This is the one dimension where the size depends on another dimension
                    self.dimensions.add(name=bounded_dim_name, size=int(self.dimensions.get('ndeplval').size / 11))
                elif bounded_dim_name == 'nobs':
                    # If this is missing it should be added with the same value as npoigages.
                    self.dimensions.add(name=bounded_dim_name, size=self.dimensions.get('npoigages').size)
                else:
                    self.__dimensions.add(bounded_dim_name)

                con.print(f'[orange3]WARNING[/]: Bounded parameter, {name}, requires dimension, {bounded_dim_name}, '
                          f'which is missing from global dimensions; '
                          f'added with size = {self.__dimensions.get(bounded_dim_name).size}')

        self.__parameters[name] = Parameter(name=name, meta=self.metadata, global_dims=self.__dimensions, verbose=self.verbose)

    def add_metadata(self, name: str,
                     metadata: dict):
        """Add a new parameter entry to the parameter metadata. This is useful for adding ad-hoc parameters.

        :param name: Name of the parameter
        :param metadata: Dictionary of metadata for the parameter
        """

        # entry_items = dict(datatype='float32', description='something new today', help='get your own help',
        #                    units='dontmatter', default=0.0, minimum=0.0, maximum=100.0, dimensions=['nhru'])
        new_entry = defaultdict(list)

        for kk, vv in metadata.items():
            new_entry[kk] = vv

        self.metadata[name] = new_entry

    def add_missing_parameters(self):
        """Add missing parameters that are required by the selected modules.
        """
        if self.verbose:
            con.print('-'*20, 'add_missing_parameters', '-'*20)   # pragma: no cover

        for cparam in self.missing_params:
            if cparam in ['nhm_deplcrv']:
                if self.verbose:   # pragma: no cover
                    con.print(f'[bold]{cparam}[/] is missing but lacks the information to be added')
                continue

            self.add(cparam)
            self.get(cparam).data = self.metadata[cparam].get('default')

            if self.verbose:   # pragma: no cover
                con.print(f'[bold]{cparam}[/] [gold3] parameter added with default value[/]')

    def add_poi(self, addl_gages: dict[str, int]):
        """Add user-specified points of interest (POIs) to the model.

        :param addl_gages: Dictionary of user-specified POIs with POI ID as key and segment index as value
        """

        poi_gage_segment = self.get('poi_gage_segment').tolist().copy()
        poi_gage_id = self.get('poi_gage_id').tolist().copy()
        poi_type = self.get('poi_type').tolist().copy()

        nsegment = self.dimensions.get('nsegment').size

        if self.verbose:
            con.print(f'Starting number of POIs: {len(poi_gage_id)}')

        for cpoi, cseg in addl_gages.items():
            if cpoi in poi_gage_id:
                idx = poi_gage_id.index(cpoi)
                con.print(f'[orange3]WARNING[/]: Existing NHM POI, {cpoi}, overridden; (was {poi_gage_segment[idx]}, now {cseg})')
                poi_gage_segment[idx] = cseg
                poi_type[idx] = 0
            elif cseg in poi_gage_segment:
                sidx = poi_gage_segment.index(cseg)
                con.print(f'[orange3]WARNING[/]: User-specified POI ({cpoi}) has same segment index ({cseg}) as existing POI ({poi_gage_id[sidx]}); replacing streamgage ID')
                poi_gage_id[sidx] = cpoi
                poi_type[sidx] = 0
            elif cseg > nsegment:
                con.print(f'[red]ERROR[/]: User-specified streamgage ({cpoi}) has segment index {cseg} which is not part of the model subset; skipping.')
            else:
                poi_gage_id.append(cpoi)
                poi_gage_segment.append(cseg)
                poi_type.append(0)
                con.print(f'Added user-specified POI streamgage ({cpoi}) at segment index {cseg}')

        if self.verbose:
            con.print(f'Final number of POIs: {len(poi_gage_id)}')
        assert len(poi_gage_id) == len(poi_gage_segment)
        assert len(poi_gage_id) == len(poi_type)

        # Update global dimensions
        for cdim in ('npoigages', 'nobs'):
            self.dimensions.get(cdim).size = len(poi_gage_id)

        for cparam in ('poi_gage_id', 'poi_gage_segment', 'poi_type'):
            # Update parameters dimensions
            self.get(cparam).dimensions.get('npoigages').size = len(poi_gage_id)

        # Update the POI parameters data
        self.get('poi_gage_id').data = np.array(poi_gage_id)
        self.get('poi_gage_segment').data = np.array(poi_gage_segment)
        self.get('poi_type').data = np.array(poi_type)

    def adjust_bounded_parameters(self):
        """Adjust the valid upper and lower values for bounded parameters.

        .. deprecated::
            Bounded parameters are now resolved at creation time in
            :meth:`Parameter.__init__`. This method will be removed in a
            future release.
        """

        warnings.warn(
            'adjust_bounded_parameters() is deprecated and will be removed in a future release. '
            'Bounded parameters are now resolved at creation time.',
            DeprecationWarning,
            stacklevel=2,
        )

        for cparam in self.parameters.values():
            cmeta = cparam.meta

            if cmeta.get('maximum') in list(self.dimensions.keys()):
                try:
                    cmeta['maximum'] = self.dimensions.get(cmeta.get('maximum')).size

                    if self.verbose:   # pragma: no cover
                        con.print(f'[bold]{cparam.name}[/]: valid upper bound adjusted to {cmeta["maximum"]}')
                except ValueError:
                    print(f'{cparam.name} has bad valid uppper bound value')
                    raise

    def resolve_units(self):
        """Adjust units metadata for parameters with initial units value of
        elev_units, precip_units, runoff_units, or temp_units.

        Some parameters have initial units metadata that are defined based on another parameter
        (e.g. elev_units, precip_units, runoff_units, or temp_units). The initial
        metadata for those parameters won't indicate the actual units
        used for the model. The value of the units metadata of any parameter
        whose initial units are one of elev_units, precip_units, runoff_units, or
        temp_units can be updated to reflect the actual units used for the model.

        """

        # Get dictionary of the *_units values
        selected_units = self.user_defined_units

        selected_units['dday/temp_units'] = f'dday {selected_units["temp_units"]}-1'
        selected_units['temp_units/elev_units'] = f'{selected_units["temp_units"]} {selected_units["elev_units"]}-1'
        selected_units['temp_units/feet'] = f'{selected_units["temp_units"]} ft-1'

        for cparam in self.parameters.values():
            cmeta = cparam.meta
            cunits = cmeta.get('units')

            if cunits in selected_units.keys():
                cmeta['units'] = selected_units[cunits]

                if self.verbose:   # pragma: no cover
                    con.print(f'{cparam.name}: {cunits} changed to {cmeta["units"]}')

    def check(self):   # pragma: no cover
        """Check all parameter variables for proper array size.
        """

        for pk in sorted(list(self.parameters.keys())):
            pp = self.get(pk)

            # print(pp.check())
            if pp.has_correct_size():
                con.print(f'[bold]{pk}[/]: Size [green4]OK[/green4]')
            else:
                con.print(f'[bold]{pk}[/]: [red]Incorrect number of values for dimensions[/red]')

            if not pp.check_values():
                pp_stats = pp.stats()
                pp_outliers = pp.outliers()
                valid_min = pp.meta['minimum']
                valid_max = pp.meta['maximum']
                default_val = pp.meta['default']

                # if not (isinstance(valid_min, str) or isinstance(valid_max, str)):
                con.print(f'    [dark_orange]WARNING[/]: Value(s) (range: {pp_stats.min}, {pp_stats.max}) outside '
                          + f'the valid range of ({valid_min}, {valid_max}); '
                          + f'under/over=({pp_outliers.under}, {pp_outliers.over})')
                # elif valid_min == 'bounded':
                #     # TODO: Handling bounded parameters needs improvement
                #     con.print(f'    [dark_orange]WARNING[/]: Bounded parameter value(s) '
                #               + f'(range: {pp_stats.min}, {pp_stats.max}) outside '
                #               + f'the valid range of ({default_val}, {valid_max})')

            dims = list(pp.dimensions.keys())

            if pp.is_scalar:
                con.print(f'    INFO: Scalar; value = {pp.data}')
            elif pp.data.ndim == 1 and np.unique(pp.data).size == 1:
                con.print(f'    INFO: dimensioned {dims}; all values are equal to {pp.data[0]}')
            elif pp.data.ndim == 2:
                if np.unique(pp.data).size == 1:
                    # All values are equal
                    con.print(f'    INFO: dimensioned {dims}; all values are equal to {pp.data[0, 0]}')
                elif (pp.data[:, 0] == pp.data.T).all():
                    # The values are equal for each month
                    con.print(f'    INFO: dimensioned {dims}; all values by {dims[1]} are equal')
                elif (pp.data[0, :] == pp.data).all():
                    # The values are equal for each hru
                    con.print(f'    INFO: dimensioned {dims}; all values by {dims[0]} are equal to {pp.data[0]}')

            if pp.name == 'snarea_curve':
                if pp.as_dataframe.values.reshape((-1, 11)).shape[0] != self.get('hru_deplcrv').unique().size:
                    con.print('  [dark_orange]WARNING[/]: snarea_curve has more entries than needed by hru_deplcrv')

    def diff(self, other: 'Parameters') -> dict:
        """A difference listing/dictionary against another Parameter object.

        :param other: Another Parameters object.
        :returns: A dictionary with keys 'self_not_other', 'other_not_self',
        and "diffs".
        """

        result: dict[Any, Any] = {}

        param_vars_self = set(self.parameters.keys())
        param_vars_other = set(other.parameters.keys())

        result['self_not_other'] = param_vars_self - param_vars_other
        result['other_not_self'] = param_vars_other - param_vars_self

        if self.verbose:   # pragma: no cover
            diffs_table = Table(title='Differences in Parameters')
            diffs_table.add_column('Parameter', justify='left', style='cyan')
            diffs_table.add_column('self', justify='left', style='magenta')
            diffs_table.add_column('other', justify='left', style='magenta')

        diffs: dict[Any, Any] = {}
        result['diffs'] = diffs
        comp_vars = param_vars_self.intersection(param_vars_other)
        for vv in sorted(list(comp_vars)):
            ss = self.get(vv).data_raw
            oo = other.get(vv).data_raw

            try:
                np.testing.assert_equal(ss, oo)
            except AssertionError:
                diffs[vv] = {"self": ss, "other": oo}

                if self.verbose:   # pragma: no cover
                    diffs_table.add_row(vv, str(ss), str(oo))

        if self.verbose and diffs_table.rows:   # pragma: no cover
            con.print(diffs_table)

        return result

    def exists(self, name: str) -> bool:
        """Checks if a parameter name exists.

        :param name: Name of the parameter
        :returns: True if parameter exists, otherwise False
        """

        return name in self

    def get(self, name: str) -> Parameter:
        """Returns a parameter object.

        :param name: The name of the parameter
        :returns: Parameter object
        """

        if self.exists(name):
            return self.__parameters[name]

        raise ParameterError(f'Parameter, {name}, does not exist.')

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Returns a pandas DataFrame for a parameter.

        If the parameter dimensions include either nhru or nsegment then the
        respective national ids (nhm_seg, nhm_id) are included, if they exist, as the index in the
        returned dataframe.

        :param name: The name of the parameter
        :returns: Pandas DataFrame of the parameter data
        """

        if not self.exists(name):
            raise KeyError(f'Parameter, {name}, does not exist')

        cparam = self.get(name)
        param_data = cparam.as_dataframe

        if cparam.is_hru_param():
            if name != 'nhm_id':
                if self.exists('nhm_id'):
                    param_id = self.get('nhm_id').as_dataframe

                    # Create a DataFrame of the parameter
                    param_data = param_data.merge(param_id, left_index=True, right_index=True)
                    param_data.set_index('nhm_id', inplace=True)
                    param_data.index.name = 'nhm_id'
            else:
                param_data = self.get('nhm_id').as_dataframe
        elif cparam.is_seg_param():
            if name != 'nhm_seg':
                if self.exists('nhm_seg'):
                    param_id = self.get('nhm_seg').as_dataframe

                    # Create a DataFrame of the parameter
                    param_data = param_data.merge(param_id, left_index=True, right_index=True)
                    param_data.set_index('nhm_seg', inplace=True)
                    param_data.index.name = 'nhm_seg'
            else:
                param_data = self.get('nhm_seg').as_dataframe
        elif name == 'snarea_curve':
            # Special handling for snarea_curve parameter
            param_data = pd.DataFrame(cparam.as_dataframe.values.reshape((-1, 11)))
            param_data.rename(columns={k: k+1 for k in param_data.columns},   # type: ignore
                              index={k: k+1 for k in param_data.index},
                              inplace=True)
            param_data.index.name = 'curve_index'
        return param_data

    def get_subset(self, name: str,
                   global_ids: list[int]) -> ParamDataRawType:
        """Returns a subset for a parameter based on the global_ids (e.g. nhm_id, nhm_seg).

        :param name: Name of the parameter
        :param global_ids: List of global IDs to extract
        :returns: Array of extracted values
        """

        param = self.get(name)
        dim_set = set(param.dimensions.keys()).intersection({'nhru', 'nssr', 'ngw', 'nsegment', 'ndeplval'})
        id_index_map: dict[Any, int] | None = {}
        cdim = dim_set.pop()

        if cdim in ['nhru', 'nssr', 'ngw', 'ndeplval']:
            # Global IDs should be in the range of nhm_id
            id_index_map = self.get('nhm_id').index_map
        elif cdim in ['nsegment']:
            # Global IDs should be in the range of nhm_seg
            id_index_map = self.get('nhm_seg').index_map

        # Zero-based indices in order of global_ids
        assert id_index_map is not None
        nhm_idx0 = [id_index_map[kk] for kk in global_ids]

        if param.dimensions.ndim == 2:
            return np.take(param.data_raw, nhm_idx0, axis=0)    # axis: 0 rows, 1 columns
        else:
            if name in ['hru_deplcrv', 'snarea_curve']:
                init_data = np.take(self.get('hru_deplcrv').data_raw, nhm_idx0, axis=0)
                # init_data = self.get('hru_deplcrv').data_raw[tuple(nhm_idx0), ]
                uniq_deplcrv = np.unique(init_data).tolist()

                match name:
                    case 'hru_deplcrv':
                        # Renumber the hru_deplcrv indices for the subset
                        uniq_dict = {xx: ii+1 for ii, xx in enumerate(uniq_deplcrv)}

                        # Create new hru_deplcrv and renumber
                        res = np.array([uniq_dict[xx] for xx in init_data])
                    case 'snarea_curve':
                        uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]
                        res = param.data_raw.reshape((-1, 11))[tuple(uniq_deplcrv0), :].reshape((-1))
                return res

            # if name == 'hru_deplcrv':
            #     # Renumber the hru_deplcrv indices for the subset
            #     uniq_dict = {xx: ii+1 for ii, xx in enumerate(uniq_deplcrv)}
            #
            #     # Create new hru_deplcrv and renumber
            #     return np.array([uniq_dict[xx] for xx in init_data])
            # elif name == 'snarea_curve':
            #     uniq_deplcrv0 = [xx - 1 for xx in uniq_deplcrv]
            #     return param.data_raw.reshape((-1, 11))[tuple(uniq_deplcrv0), :].reshape((-1))
            else:
                # All other 1D arrays
                return np.take(param.data_raw, nhm_idx0, axis=0)    # axis: 0 rows, 1 columns

    def outlier_ids(self, name: str) -> list[int]:
        """Returns list of HRU or segment IDs of invalid parameter values

        :param name: Name of the parameter
        :returns: List of HRU or segment IDs
        """

        cparam = self.get(name)

        param_data = self.get_dataframe(name)
        bad_value_ids = param_data[(param_data[name] < cparam.meta['minimum']) |
                                   (param_data[name] > cparam.meta['maximum'])].index.tolist()

        return bad_value_ids

    def parameters_info(self) -> pd.DataFrame:
        """Returns a dataframe of the parameter metadata information.

        :returns: DataFrame
        """

        out_list = []

        if self.__control is not None:
            modules_used = set(self.__control.modules.values()).union(set(self.__control.additional_modules))

        for pk in sorted(list(self.parameters.keys())):
            pp = self.get(pk)
            md = pp.meta

            param_modules = md.get('modules')
            assert type(param_modules) is list

            if self.__control is not None:
                module_list = list(modules_used.intersection(set(param_modules)))

                if len(module_list) == 0:
                    # We have a parameter that is not needed by the declared modules
                    if self.verbose:   # pragma: no cover
                        print(f'{pp.name} not used with selected modules')
                        print(f'    {md.get("modules")}')
                    continue
            else:
                module_list = param_modules

            for idx, mm in enumerate(module_list):
                if mm in external_module_map:
                    module_list[idx] = external_module_map[mm]

            modules = ', '.join(module_list)

            dim_list = list(pp.dimensions.keys())

            if pk in ['gwflow_coef', 'gwsink_coef', 'gwstor_init', 'gwstor_min', 'gw_seep_coef']:
                dim_list[0] = 'ngw'
            elif pk in ['ssr2gw_exp', 'ssr2gw_rate', 'ssstor_init', 'ssstor_init_frac']:
                dim_list[0] = 'nssr'

            dims = ', '.join(dim_list)

            try:
                act_min = pp.data_raw.min()
                act_max = pp.data_raw.max()
            except np.core._exceptions.UFuncTypeError:   # type: ignore
                act_min = ''
                act_max = ''

            out_list.append([pp.name, md.get('description', ''),
                             md.get('datatype', ''),
                             md.get('units', ''),
                             md.get('default'),
                             md.get('minimum'),
                             md.get('maximum'),
                             act_min,
                             act_max,
                             dims,
                             modules])

        col_names = ['parameter_name', 'description', 'datatype', 'units',
                     'default', 'valid_minimum', 'valid_maximum', 'actual_minimum',
                     'actual_maximum', 'dimension', 'modules']

        return pd.DataFrame.from_records(out_list, columns=col_names)

    def plot(self, name: str,
             output_dir: str | None = None,
             limits: LimitOptions | list[float] | tuple[float, float] | None = 'absolute',
             mask_defaults: str | None = None,
             **kwargs):   # pragma: no cover
        """Plot a parameter.

        Plots either to the screen or an output directory.

        :param name: Name of parameter to plot
        :param output_dir: Directory to write plot to (None for write to screen only)
        :param limits: Limits to use for colorbar. One of 'valid', 'centered', 'absolute', or list of floats.
        :param mask_defaults: Color for defaults values
        """

        # is_monthly = False
        time_index = None

        if self.exists(name):
            cparam = self.get(name)

            if set(cparam.dimensions.keys()).intersection({'nmonths'}):
                # Need 12 monthly plots of parameter
                # is_monthly = True
                time_index = 0  # starting time index
                param_data = self.get_dataframe(name).iloc[:, time_index].to_frame(name=name)
            else:
                param_data = self.get_dataframe(name).iloc[:]

            if mask_defaults is not None:
                param_data = param_data.mask(param_data == cparam.meta['default'])

            if isinstance(limits, str):
                if limits == 'valid':
                    valid_min = cparam.meta['minimum']
                    valid_max = cparam.meta['maximum']

                    # Use the defined valid range of possible values
                    if valid_min == 'bounded':
                        # Parameters with bounded values need to always use the actual range of values
                        drange = [cparam.data_raw.min().min(), cparam.data_raw.max().max()]
                    elif name == 'jh_coef':
                        drange = [-0.05, 0.05]
                    else:
                        drange = [valid_min, valid_max]
                elif limits == 'centered':
                    # Use the maximum range of the actual data values
                    lim = max(abs(cparam.data_raw.min().min()), abs(cparam.data_raw.max().max()))
                    drange = [-lim, lim]
                elif limits == 'absolute':
                    # Use the min and max of the data values
                    drange = [cparam.data_raw.min().min(), cparam.data_raw.max().max()]
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
            kwargs.pop('cmap', None)

            if mask_defaults is not None:
                cmap.set_bad(mask_defaults, 0.7)

            if set(cparam.dimensions.keys()).intersection({'nhru', 'ngw', 'nssr'}):
                # Get extent information
                if self.__hru_poly is not None:
                    minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds

                    crs_proj = get_projection(self.__hru_poly)

                    # Takes care of multipolygons that are in the NHM geodatabase/shapefile
                    geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)

                    df_mrg = geoms_exploded.merge(param_data, left_on=self.__hru_shape_key,
                                                  right_index=True, how='left')

                    fig_width, fig_height = get_figsize([minx, maxx, miny, maxy], **dict(kwargs))
                    kwargs.pop('init_size', None)

                    fig = plt.figure(figsize=(fig_width, fig_height))

                    ax = plt.axes(projection=crs_proj)

                    try:
                        ax.coastlines()
                    except AttributeError:
                        pass

                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                    gl.top_labels = None
                    gl.right_labels = None
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER

                    ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

                    if time_index is not None:
                        plt.title(f'Variable: {name},  Month: {time_index+1}')
                    else:
                        plt.title(f'Variable: {name}')

                    if mask_defaults is not None:
                        plt.annotate(f'NOTE: Values = {cparam.meta["default"]} are masked', xy=(0.5, 0.01),
                                     xycoords='axes fraction', va='center', ha='center',
                                     fontsize=10, fontweight='bold',
                                     bbox=dict(boxstyle="round", facecolor=mask_defaults, alpha=1.0))

                    # Setup the color bar
                    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                    mapper.set_array(df_mrg[name])
                    cax = fig.add_axes((ax.get_position().x1 + 0.01,
                                        ax.get_position().y0,
                                        0.02,
                                        ax.get_position().height))

                    # TODO: 2022-06-17 PAN - Categorical variables require entry in two places.
                    #       The first place is here for labelling and the second place is in
                    #       plot_helpers.py for adjusting the color boundaries. Need to figure out
                    #       a more consistent/better way to do this.
                    if name == 'hru_deplcrv':
                        # tck_arr = np.arange(param_data.min().min(), param_data.max().max()+1)
                        tck_arr = np.arange(drange[0], drange[1]+1)
                        cb = plt.colorbar(mapper, cax=cax, ticks=tck_arr, label='Curve index')
                        cb.ax.tick_params(length=0)
                    elif name == 'soil_type':
                        tck_arr = np.arange(drange[0], drange[1]+1)
                        cb = plt.colorbar(mapper, cax=cax, ticks=tck_arr, label=name)
                        cb.ax.tick_params(length=0)
                    elif name == 'calibration_status':
                        tck_arr = np.arange(drange[0], drange[1]+1)
                        cb = plt.colorbar(mapper, cax=cax, ticks=tck_arr, label=name)
                        cb.ax.tick_params(length=0)
                    else:
                        plt.colorbar(mapper, cax=cax, label=cparam.meta['units'])

                    col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
                                                  cmap=cmap, norm=norm, **dict(kwargs))
                    # col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[name],
                    #                               **dict(kwargs, cmap=cmap, norm=norm))

                    if output_dir is not None:
                        if time_index is not None:
                            # First month
                            plt.savefig(f'{output_dir}/{name}_{time_index+1:02}.png', dpi=150, bbox_inches='tight')

                            for tt in range(1, 12):
                                # Months 2 through 12
                                param_data = self.get_dataframe(name).iloc[:, tt].to_frame(name=name)

                                if mask_defaults is not None:
                                    param_data = param_data.mask(param_data == cparam.meta['default'])

                                df_mrg = geoms_exploded.merge(param_data, left_on=self.__hru_shape_key,
                                                              right_index=True, how='left')

                                ax.set_title(f'Variable: {name},  Month: {tt+1}')
                                col.set_array(df_mrg[name])
                                plt.savefig(f'{output_dir}/{name}_{tt+1:02}.png', dpi=150, bbox_inches='tight')
                        else:
                            plt.savefig(f'{output_dir}/{name}.png', dpi=150, bbox_inches='tight')

                        # Close the figure so we don't chew up memory
                        fig.clf()
                        plt.close()
                        gc.collect()
            elif set(cparam.dimensions.keys()).intersection({'nsegment'}):
                # Plot segment-related parameters
                if self.__seg_poly is not None:
                    if self.__hru_poly is not None:
                        minx, miny, maxx, maxy = self.__hru_poly.geometry.total_bounds
                    else:
                        minx, miny, maxx, maxy = self.__seg_poly.geometry.total_bounds

                    seg_geoms_exploded = self.__seg_poly.explode(index_parts=True).reset_index(level=1, drop=True)

                    crs_proj = get_projection(self.__seg_poly)

                    df_mrg = seg_geoms_exploded.merge(param_data, left_on=self.__seg_shape_key,
                                                      right_index=True, how='left')

                    fig_width, fig_height = get_figsize([minx, maxx, miny, maxy], **dict(kwargs))
                    kwargs.pop('init_size', None)

                    fig = plt.figure(figsize=(fig_width, fig_height))

                    ax = plt.axes(projection=crs_proj)

                    try:
                        ax.coastlines()
                    except AttributeError:
                        pass

                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
                    gl.top_labels = None
                    gl.right_labels = None
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER

                    ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)

                    plt.title('Variable: {}'.format(name))

                    if kwargs.get('vary_color', True):
                        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                        mapper.set_array(df_mrg[name])
                        cax = fig.add_axes((ax.get_position().x1 + 0.01,
                                            ax.get_position().y0, 0.02,
                                            ax.get_position().height))
                        plt.colorbar(mapper, cax=cax, label=cparam.meta['units'])   # , shrink=0.6

                    if self.__hru_poly is not None:
                        hru_geoms_exploded = self.__hru_poly.explode(index_parts=True).reset_index(level=1, drop=True)
                        hru_poly = plot_polygon_collection(ax, hru_geoms_exploded.geometry,
                                                           cmap=cmap, norm=norm,
                                                           **dict(kwargs, linewidth=0.5, alpha=0.7))

                    col = plot_line_collection(ax, df_mrg.geometry, values=df_mrg[name], cmap=cmap,
                                               norm=norm, **dict(kwargs))

                    if mask_defaults is not None:
                        plt.annotate(f'NOTE: Values = {cparam.meta["default"]} are masked', xy=(0.5, 0.01),
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

    def poi_upstream_hrus(self, poi: str | list[str] | KeysView) -> dict[str, list[int]]:
        """Returns a dictionary of POI to upstream global HRU IDs.

        :param poi: POI ID or list of POI IDs

        :returns: Dictionary of POI to upstream global HRU IDs
        """

        if isinstance(poi, str):
            poi = [poi]
        elif isinstance(poi, KeysView):
            poi = list(poi)

        poi_hrus = {}
        nhm_seg = self.get('nhm_seg').data_raw
        assert type(nhm_seg) is np.ndarray and nhm_seg.dtype == np.int32
        poi_seg0_dict = self.poi_to_seg0

        # Generate stream network for the model
        dag_streamnet = self.stream_network()

        for cpoi in poi:
            # Lookup global segment id for the current POI
            cseg = poi_seg0_dict[cpoi]
            dsmost_seg = [nhm_seg[cseg].item()]

            poi_hrus[cpoi] = self._upstream_hrus(dag_streamnet, dsmost_seg)

        return poi_hrus

    def poi_upstream_segments(self, poi: str | list[str] | KeysView) -> dict[str, list[int]]:
        """Returns a dictionary of POI to upstream global segment IDs.

        :param poi: POI ID or list of POI IDs

        :returns: Dictionary of POI to upstream global segment IDs
        """

        if isinstance(poi, str):
            poi = [poi]
        elif isinstance(poi, KeysView):
            poi = list(poi)

        poi_segs = {}
        nhm_seg = self.get('nhm_seg').data_raw
        assert type(nhm_seg) is np.ndarray
        pois_dict = self.poi_to_seg0

        # Generate stream network for the model
        dag_streamnet = self.stream_network()

        for cpoi in poi:
            # Lookup global segment id for the current POI
            dsmost_seg = [nhm_seg[pois_dict[cpoi]].item()]

            poi_segs[cpoi] = self._upstream_segments(dag_streamnet, dsmost_seg)

        return poi_segs

    def remove(self, name: str | Sequence[str] | set[str]):
        """Delete one or more parameters if they exist.

        :param name: parameter or list of parameters to remove
        """

        if isinstance(name, str):
            name = [name]

        for cparam in name:
            if self.exists(cparam):
                del self.__parameters[cparam]

                if self.verbose:   # pragma: no cover
                    con.print(f'[bold]{cparam}[/] [gold3]parameter removed[/]')

    def remove_poi(self, poi: str | list[str]):
        """Remove POIs by gage_id.

        :param poi: POI id to remove
        """

        if isinstance(poi, str):
            poi = [poi]

        # First get array of poi_gage_id indices matching the specified POI IDs
        poi_ids = self.get('poi_gage_id').data_raw.tolist()
        assert type(poi_ids) is list
        poi_del_indices = []
        for xx in poi:
            # We silently ignore missing POIs
            if xx in poi_ids:
                poi_del_indices.append(poi_ids.index(xx))  # type: ignore

        # poi_ids = self.get('poi_gage_id').data
        # sorter = np.argsort(poi_ids)
        # poi_del_indices = sorter[np.searchsorted(poi_ids, poi, sorter=sorter)]

        poi_parameters = ['poi_gage_id', 'poi_gage_segment', 'poi_type']

        if len(poi_del_indices) > 0:
            if self.get('poi_gage_id').dimensions.get('npoigages').size == len(poi_del_indices):
                # We're trying to remove all the POIs
                for pp in poi_parameters:
                    self.remove(pp)

                self.dimensions.get('npoigages').size = 0
            else:
                # Remove the matching poi gage entries from each of the poi-related parameters
                for pp in poi_parameters:
                    self.get(pp).remove_by_index('npoigages', poi_del_indices)

                # Update the global npoigages dimension
                self.dimensions.get('npoigages').size -= len(poi_del_indices)
                self.dimensions.get('nobs').size -= len(poi_del_indices)

    def segment_upstream_hrus(self, segs: int | list[int] | KeysView | npt.NDArray) -> dict[int, list[int]]:
        """Returns a dictionary of segment to upstream global HRU IDs.

        :param segs: Global segment ID or list of global segment IDs

        :returns: Dictionary of global segment ID to upstream global HRU IDs
        """

        if isinstance(segs, int):
            segs = [segs]
        elif isinstance(segs, KeysView):
            segs = list(segs)
        elif isinstance(segs, np.ndarray):
            segs = segs.tolist()  # type: ignore

        assert type(segs) is list

        seg_hrus = {}

        # Generate stream network for the model
        dag_streamnet = self.stream_network()

        for cseg in segs:
            # Lookup segment for the current POI
            dsmost_seg = [cseg]

            seg_hrus[cseg] = self._upstream_hrus(dag_streamnet, dsmost_seg)

        return seg_hrus

    def segment_upstream_segments(self, segs: int | list[int] | KeysView | npt.NDArray) -> dict[int, list[int]]:
        """Returns a dictionary of global segment IDs to upstream global segment IDs.

        :param segs: global segment IDs or list of global segment IDs

        :returns: Dictionary of global segment ID to upstream global segment IDs
        """

        if isinstance(segs, int):
            segs = [segs]
        elif isinstance(segs, KeysView):
            segs = list(segs)
        elif isinstance(segs, np.ndarray):
            segs = segs.tolist()

        assert type(segs) is list

        us_segs = {}

        # Generate stream network for the model
        dag_streamnet = self.stream_network()

        for cseg in segs:
            # Lookup segment for the current segment
            dsmost_seg = [cseg]
            us_segs[cseg] = self._upstream_segments(dag_streamnet, dsmost_seg)

        return us_segs

    def shapefile_hrus(self, filename: str,
                       layer_name: str | None = None,
                       shape_key: str | None = None):   # pragma: no cover
        """Read a shapefile or geodatabase that corresponds to HRUs.

        :param filename: name of shapefile or geodatabase
        :param layer_name: name of layer in geodatabase
        :param shape_key: name of attribute for key
        """

        self.__hru_poly = geopandas.read_file(filename, layer=layer_name)

        if self.__hru_poly.crs.name == 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version':   # type: ignore
            print('Overriding USGS aea crs with EPSG:5070')
            self.__hru_poly.crs = 'EPSG:5070'   # type: ignore
        # elif self.__hru_poly.crs.name[0:5] == 'NAD83':   # type: ignore
        #     self.__hru_poly.to_crs('epsg:4326', inplace=True)

        self.__hru_shape_key = shape_key

    def shapefile_segments(self, filename: str,
                           layer_name: str | None = None,
                           shape_key: str | None = None):   # pragma: no cover
        """Read a shapefile or geodatabase that corresponds to stream segments.

        :param filename: name of shapefile or geodatabase
        :param layer_name: name of layer in geodatabase
        :param shape_key: name of attribute for key
        """

        self.__seg_poly = geopandas.read_file(filename, layer=layer_name)

        if self.__seg_poly.crs.name == 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version':   # type: ignore
            print('Overriding USGS aea crs with EPSG:5070')
            self.__seg_poly.crs = 'EPSG:5070'   # type: ignore
        # elif self.__seg_poly.crs.name[0:5] == 'NAD83':   # type: ignore
        #     self.__seg_poly.to_crs('epsg:4326', inplace=True)

        self.__seg_shape_key = shape_key

    def stream_network(self, tosegment: str = 'tosegment_nhm',
                       seg_id: str = 'nhm_seg') -> nx.DiGraph | None:
        """Create Directed, Acyclic Graph (DAG) of stream network.

        :param tosegment: name of parameter to use for HRU tosegment
        :param seg_id: name of parameter to use for the segment IDs
        :returns: directed-acyclic-graph (DAG)
        """

        seg = self.get(seg_id).tolist()
        toseg = self.get(tosegment).tolist()
        assert type(seg) is list and type(toseg) is list

        dag_ds = nx.DiGraph()
        for ii, vv in enumerate(toseg):
            #     dag_ds.add_edge(ii+1, vv)
            if vv == 0:
                dag_ds.add_edge(seg[ii], 'Out_{}'.format(seg[ii]))
            else:
                dag_ds.add_edge(seg[ii], vv)

        return dag_ds

    def update_element(self, name: str,
                       id1: int,
                       value: int | float | list[int] | list[float]):
        """Update single value or row of values (e.g. nhru by nmonths) for a
        given nhm_id, nhm_seg, or 0 (for scalars).

        :param name: name of parameter to update
        :param id1: scalar nhm_id or nhm_seg
        :param value: new value(s)
        """

        # NOTE: id1 is either an nhm_id or nhm_seg (both are 1-based)
        cparam = self.get(name)
        idx0 = 0

        try:
            if cparam.is_hru_param():
                # Lookup index for nhm_id
                idx0 = self.get('nhm_id')._value_index_1d(id1).item()
            elif cparam.is_seg_param():
                # Lookup index for nhm_seg
                idx0 = self.get('nhm_seg')._value_index_1d(id1).item()
            elif cparam.is_scalar:
                idx0 = 0
        except ValueError:
            raise ValueError(f'Exactly one index should be found for {id1}')

        # This will raise a ValueError if more than one index is found
        cparam.update_element(idx0, value)

    def write_dimensions_xml(self, output_dir: str):
        """Write global dimensions.xml file.

        :param output_dir: output path for dimensions.xml file
        """

        # Write the global dimensions xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_dimensions)).toprettyxml(indent='    ')
        with open(f'{output_dir}/{DIMENSIONS_XML}', 'w') as ff:
            ff.write(xmlstr)

    def write_paramdb(self, output_dir: str):
        """Write all parameters using the paramDb output format.

        :param output_dir: output path for paramDb files
        """

        # Check for / create output directory
        try:
            if self.verbose:   # pragma: no cover
                print(f'Creating output directory: {output_dir}')
            os.makedirs(output_dir)
        except OSError:
            if self.verbose:   # pragma: no cover
                print("\tUsing existing directory")

        # Write the global dimensions xml file
        self.write_dimensions_xml(output_dir)

        # Write the global parameters xml file
        self.write_parameters_xml(output_dir)

        for xx in self.parameters.values():
            # Write out each parameter in the paramDb csv format
            if self.verbose:   # pragma: no cover
                print(xx.name)

            with open(f'{output_dir}/{xx.name}.csv', 'w') as ff:
                ff.write(xx.toparamdb())

    def write_parameter_file(self, filename: str | Path,
                             header: list[str] | None = None):
        """Write a PRMS parameter file.

        :param filename: name of parameter file
        :param header: list of header lines
        """

        if isinstance(filename, str):
            filename = Path(filename)

        # Write the parameters out to a file
        outfile = open(str(filename), 'w')

        if header is not None:
            if len(header) > 2:
                # TODO: 2023-04-19 - this check should happen before
                #       opening the output file.
                raise ValueError('Header should be a list of two items')
            if len(header) == 1:
                # Must have two header lines
                outfile.write('Written by pyPRMS\n')

            for hh in header:
                # Write out any header stuff
                outfile.write(f'{hh}\n')
        else:
            # Write a default header
            outfile.write('Written by pyPRMS\n')
            outfile.write('Comment: It is all downhill from here\n')

        # Dimension section must be written first
        outfile.write(f'{CATEGORY_DELIM} Dimensions {CATEGORY_DELIM}\n')

        for (kk, vv) in self.dimensions.items():
            # Write each dimension name and size separated by VAR_DELIM
            outfile.write(f'{VAR_DELIM}\n')
            outfile.write(f'{kk}\n')
            outfile.write(f'{vv.size:d}\n')

        if self.prms_version < PRMS6_DEV_VERSION and {'ngw', 'nssr'}.isdisjoint(set(self.dimensions.keys())):
            # Add the ngw and nssr dimensions. These are always equal to nhru.
            for kk in ['ngw', 'nssr']:
                outfile.write(f'{VAR_DELIM}\n')
                outfile.write(f'{kk}\n')
                outfile.write(f'{self.dimensions["nhru"].size:d}\n')

        # Now write out the Parameter category
        order = ['name', 'dimensions', 'datatype', 'data']

        outfile.write(f'{CATEGORY_DELIM} Parameters {CATEGORY_DELIM}\n')

        for vv in self.parameters.values():
            datatype = PTYPE_TO_PRMS_TYPE[vv.meta.get('datatype')]

            for item in order:
                # Write each variable out separated by self.__rowdelim
                if item == 'dimensions':
                    # Write number of dimensions first
                    outfile.write(f'{vv.dimensions.ndim}\n')

                    for dd in vv.dimensions.values():
                        # Write dimension names
                        if self.prms_version < PRMS6_DEV_VERSION:
                            # On-the-fly change of dimension names for certain parameters
                            # when the prms version is not an experimental development version.
                            if dd.name == 'nhru':
                                if vv.name in ['gwflow_coef', 'gwsink_coef', 'gwstor_init',
                                               'gwstor_min', 'gw_seep_coef']:
                                    outfile.write('ngw\n')
                                elif vv.name in ['ssr2gw_exp', 'ssr2gw_rate', 'ssstor_init',
                                                 'ssstor_init_frac']:
                                    outfile.write('nssr\n')
                                else:
                                    outfile.write(f'{dd.name}\n')
                            else:
                                outfile.write(f'{dd.name}\n')
                        else:
                            outfile.write(f'{dd.name}\n')
                elif item == 'datatype':
                    # dimsize (which is computed) must be written before datatype
                    outfile.write(f'{vv.data_raw.size}\n')
                    outfile.write(f'{datatype}\n')
                elif item == 'data':
                    # Write one value per line
                    # WARNING: 2019-10-10: had to change next line from order='A' to order='F'
                    #          because flatten with 'A' was only honoring the Fortran memory layout
                    #          if the array was contiguous which isn't always the
                    #          case if the arrays have been altered in size.
                    for xx in vv.data_raw.ravel(order='F'):
                        if vv.meta.get('datatype', 'null') in ['float32', 'float64']:
                            # Float and double types have to be formatted specially so
                            # they aren't written in exponential notation or with
                            # extraneous zeroes
                            tmp = f'{xx:<20.7f}'.rstrip('0 ')
                            if tmp[-1] == '.':
                                tmp += '0'

                            outfile.write(f'{tmp}\n')
                        else:
                            outfile.write(f'{xx}\n')
                elif item == 'name':
                    # Write the self.__rowdelim before the variable name
                    outfile.write(f'{VAR_DELIM}\n')
                    outfile.write(f'{vv.name}\n')

        outfile.close()

    def write_parameter_netcdf(self, filename: str | Path):
        """Write parameters to a netcdf format file.

        :param filename: full path for output file
        """

        if isinstance(filename, str):
            filename = Path(filename)

        # Update units metadata for parameters with units of
        # elev_units, precip_units, runoff_units, or temp_units
        self.resolve_units()

        # Create the netcdf file
        nc_hdl = nc.Dataset(filename, 'w', clobber=True)

        # Create dimensions
        for (kk, vv) in self.dimensions.items():
            if kk != 'one':
                # Dimension 'one' is only used for scalars in PRMS
                nc_hdl.createDimension(kk, vv.size)

        # Create the variables
        # hruo = nco.createVariable('hru', 'i4', ('hru'))
        for vv in self.parameters.values():
            curr_datatype = NETCDF_DATATYPES[PTYPE_TO_PRMS_TYPE[vv.meta.get('datatype')]]

            if curr_datatype != 'S1':
                if list(vv.dimensions.keys())[0] == 'one':
                    # Scalar values
                    curr_param = nc_hdl.createVariable(vv.name, curr_datatype,
                                                       fill_value=nc.default_fillvals[curr_datatype], zlib=True)
                else:
                    curr_param = nc_hdl.createVariable(vv.name, curr_datatype,
                                                       tuple(list(vv.dimensions.keys())[::-1]),
                                                       fill_value=nc.default_fillvals[curr_datatype], zlib=True)

                # Add the attributes
                for ck, cmeta in vv.meta.items():
                    if ck in ['minimum', 'maximum']:
                        curr_param.setncattr(f'valid_{ck}', cmeta)
                    elif ck in ['help', 'description', 'units']:
                        curr_param.setncattr(ck, cmeta)

                # Write the data
                if len(vv.dimensions.keys()) == 1:
                    curr_param[:] = vv.data
                elif len(vv.dimensions.keys()) == 2:
                    curr_param[:, :] = vv.data.transpose()
            else:
                # String parameter
                # Get the maximum string length in the array of data
                str_size = len(max(vv.data, key=len))

                # Create a dimension for the string length
                nc_hdl.createDimension(vv.name + '_nchars', str_size)

                # Temporary to add extra dimension for number of characters
                tmp_dims = list(vv.dimensions.keys())
                tmp_dims.extend([vv.name + '_nchars'])
                curr_param = nc_hdl.createVariable(vv.name, curr_datatype, tuple(tmp_dims),
                                                   fill_value=nc.default_fillvals[curr_datatype], zlib=True)

                # Add the attributes
                for ck, cmeta in vv.meta.items():
                    if ck in ['help', 'description', 'units']:
                        curr_param.setncattr(ck, cmeta)

                # Write the data
                if len(tmp_dims) == 1:
                    curr_param[:] = nc.stringtochar(vv.data)
                elif len(tmp_dims) == 2:
                    # curr_param._Encoding = 'ascii'

                    # The stringtochar() routine won't handle the unicode numpy
                    # datatype properly so we force it to dtype='S'
                    curr_param[:, :] = nc.stringtochar(vv.data.astype('S'))
            sys.stdout.flush()
        # Close the netcdf file
        nc_hdl.close()

    def write_parameters_metadata_csv(self, filename: str,
                                      sep: str = '\t'):
        """Writes the parameter metadata to a CSV file.

        :param filename: output filename
        :param sep: separator character
        """

        df = self.parameters_info()

        if sep == ',':
            df.to_csv(filename, sep=sep, quotechar='"', index=False)
        else:
            df.to_csv(filename, sep=sep, index=False)

    def write_parameters_xml(self, output_dir: str):
        """Write global parameters.xml file.

        :param output_dir: output path for parameters.xml file
        """

        # Write the global parameters xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_parameters)).toprettyxml(indent='    ')
        with open(f'{output_dir}/{PARAMETERS_XML}', 'w') as ff:
            ff.write(xmlstr)

    def _condition_check_ctl(self, cstr: str) -> bool:
        """Takes a string of the form '<control_var> <op> <value>' and checks
        if the condition is True.

        :param cstr: string of the form '<control_var> <op> <value>'
        :returns: True if the condition is met, False otherwise
        """

        assert self.control is not None
        var, op, value = cstr.split(' ')

        cdtype = NEW_PTYPE_TO_DTYPE[self.control.get(var).meta['datatype']]
        ctl_val = self.control.get(var).values
        assert ctl_val is not None
        value = cdtype(value)

        return cond_check[op](ctl_val, value)

    def _condition_check_dim(self, cstr: str) -> bool:
        """Takes a string of the form '<dimension> <op> <value>' and checks
        if the condition is True.

        :param cstr: string of the form '<dimension> <op> <value>'
        :returns: True if the condition is met, False otherwise
        """

        var, op, value = cstr.split(' ')
        value = int(value)  # type: ignore

        if self.dimensions.exists(var):
            return cond_check[op](self.dimensions.get(var).size, value)
        return False

    def _read(self):
        """Abstract function for reading parameters into Parameters object.
        """

        assert False, 'Parameters._read() must be defined by child class'

    def _required_parameters(self) -> set:
        """Return set of parameters required by modules selected in control file.

        :returns: set of required parameter names
        """
        if self.__control is None:
            # TODO: 20230727 PAN - this should raise an exception
            return set()

        modules_used = set(self.__control.modules.values()).union(set(self.__control.additional_modules))

        # -------------------------------
        # Get set of parameters required by the modules used
        pset = set()
        for kk, vv in self.metadata.items():
            for mm in vv['modules']:
                if mm in modules_used:
                    pset.add(kk)

        # Remove parameters that do not meet secondary requirements defined
        # in the metadata
        return self._trim_req_params(pset)

    def _trim_req_params(self, param_set: set) -> set:
        """Remove parameters from a set of parameters that do not meet secondary requirements.

        :param param_set: set of parameter names
        :returns: set of parameter names that meet secondary requirements
        """

        remove_set = set()

        for cparam in param_set:
            for xx in self.metadata[cparam].get('requires_control', []):
                if not self._condition_check_ctl(xx):
                    remove_set.add(cparam)
                    if self.verbose:   # pragma: no cover
                        con.print(f'[bold]{cparam}[/]: Control condition ({xx}) not met')

            for xx in self.metadata[cparam].get('requires_dimension', []):
                if not self._condition_check_dim(xx):
                    remove_set.add(cparam)
                    if self.verbose:   # pragma: no cover
                        con.print(f'[bold]{cparam}[/]: Dimension condition ({xx}) not met')

        for vv in remove_set:
            param_set.remove(vv)

        return param_set

    def _upstream_hrus(self, streamnet: nx.DiGraph,
                       dsmost_seg: list[int]) -> list[int]:
        """Get list of HRUs that contribute to the given stream segments.

        :param streamnet: Directed, Acyclic Graph (DAG) of stream network
        :param dsmost_seg: list of downstream-most segment IDs

        :returns: list of HRUs that contribute to the stream segments
        """

        # Get subset of stream network for given segment
        dag_ds_subset = get_streamnet_subset(streamnet, [], dsmost_seg)

        # Create list of segments in the subset
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

        # Build list of HRUs that contribute to the POI
        final_hru_list = []

        for xx in toseg_idx:
            try:
                for yy in self.seg_to_hru[xx]:
                    final_hru_list.append(yy)
            except KeyError:
                if self.verbose:
                    # Not all segments have HRUs connected to them
                    print(f'Segment {xx} has no HRUs connected to it')
                pass

        final_hru_list.sort()
        return final_hru_list

    def _upstream_segments(self, streamnet: nx.DiGraph,
                           dsmost_seg: list[int]) -> list[int]:
        """Get list of segments that contribute to the given stream segments.

        :param streamnet: Directed, Acyclic Graph (DAG) of stream network
        :param dsmost_seg: list of downstream-most segment IDs

        :returns: list of segments that contribute to the stream segments
        """

        # Get subset of stream network for given POI
        dag_ds_subset = get_streamnet_subset(streamnet, [], dsmost_seg)

        # Create list of segments in the subset
        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

        return toseg_idx

    def _resize_dims(self,
                     num_hru: int,
                     num_seg: int,
                     num_deplcrv: int,
                     num_poi: int) -> dict[str, int]:
        """Return a dictionary of dimensions resized for a model subset.

        :param num_hru: Number of HRUs in the model subset
        :param num_seg: Number of segments in the model subset
        :param num_deplcrv: Number of snow depletion curves in the model subset
        :param num_poi: Number of Points-of-Interest (POIs) in the model subset
        :returns: Dictionary of the resized dimensions
        """

        dims = {kk.name: kk.size for kk in self.dimensions.values()}

        for dd in list(dims.keys()):
            if dd in HRU_DIMS:
                dims[dd] = num_hru
            elif dd == 'nsegment':
                dims[dd] = num_seg
            elif dd == 'ndeplval':
                dims[dd] = num_deplcrv * 11
                dims['ndepl'] = num_deplcrv
            elif dd == 'npoigages':
                dims[dd] = num_poi
                dims['nobs'] = num_poi

        return dims

    def create_subset(self,
                      hru_order_subset: list[int],
                      new_hru_segment: list[int],
                      new_nhm_seg: list[int],
                      new_poi_gage_id: list[str],
                      new_poi_gage_segment: list[int],
                      new_poi_type: list[int],
                      new_tosegment: list[int]) -> 'Parameters':
        """Create a new Parameters object that is a subset of this parameter database.

        :param hru_order_subset: List of HRUs to include in the subset
        :param new_hru_segment: List of HRU segments for the HRU subset
        :param new_nhm_seg: List of NHM segments for the subset
        :param new_poi_gage_id: List of POI gage IDs for the subset
        :param new_poi_gage_segment: List of POI gage segments for the subset
        :param new_poi_type: List of POI types for the subset
        :param new_tosegment: List of tosegment values for the subset
        :returns: New Parameters object with the subset of parameters
        """

        # Get unique depletion curves referenced by the HRU subset
        hru_deplcrv_subset = self.get_subset('hru_deplcrv', hru_order_subset)
        uniq_deplcrv: list = np.unique(hru_deplcrv_subset).tolist()

        # Build parameter list, removing POI params if no POIs exist
        params = list(self.parameters.keys())

        if len(new_poi_gage_segment) == 0:
            con.print('[gold3]WARNING[/]: No POIs found for model subset')
            for rp in ['poi_gage_id', 'poi_gage_segment', 'poi_type']:
                if rp in params:
                    params.remove(rp)

        params.sort()

        # Resize dimensions for the subset
        dims = self._resize_dims(num_hru=len(hru_order_subset),
                                 num_seg=len(new_nhm_seg),
                                 num_deplcrv=len(uniq_deplcrv),
                                 num_poi=len(new_poi_gage_segment))

        # Build the new Parameters instance
        new_ps = Parameters(metadata=self.__full_metadata)

        for dd, dv in dims.items():
            new_ps.dimensions.add(dd, dv)

        for pp in params:
            src_param = self.get(pp)

            new_ps.add(name=pp)
            cnew_param = new_ps.get(pp)

            ndims = src_param.ndim
            dim_order = list(src_param.dimensions.keys())
            first_dimension = dim_order[0]
            outdata = None

            if ndims == 0:
                # Scalar parameters
                outdata = src_param.data
            elif ndims == 1:
                if first_dimension == 'nsegment':
                    if pp in ['tosegment']:
                        outdata = np.array(new_tosegment)
                    else:
                        outdata = self.get_subset(pp, new_nhm_seg)
                elif first_dimension == 'ndeplval':
                    # snarea_thresh - stored in C-order unlike other 2D arrays
                    outdata = self.get_subset(pp, hru_order_subset)
                elif first_dimension == 'npoigages':
                    if pp == 'poi_gage_segment':
                        outdata = np.array(new_poi_gage_segment)
                    elif pp == 'poi_gage_id':
                        outdata = np.array(new_poi_gage_id)
                    elif pp == 'poi_type':
                        outdata = np.array(new_poi_type)
                    else:
                        con.print(f'[red]ERROR[/]: Unknown parameter, {pp}, with dimension {first_dimension}')
                elif first_dimension in HRU_DIMS:
                    if pp == 'hru_deplcrv':
                        outdata = self.get_subset(pp, hru_order_subset)
                    elif pp == 'hru_segment':
                        outdata = np.array(new_hru_segment)
                    else:
                        outdata = self.get_subset(pp, hru_order_subset)
                else:
                    con.print(f'[red]ERROR[/]: No rules to handle dimension {first_dimension}')
            elif ndims == 2:
                if first_dimension == 'nsegment':
                    outdata = self.get_subset(pp, new_nhm_seg)
                elif first_dimension in HRU_DIMS:
                    outdata = self.get_subset(pp, hru_order_subset)
                else:
                    con.print(f'[red]ERROR[/]: No rules to handle 2D parameter, {pp}, '
                              f'with dimension {first_dimension}')

            cnew_param.data = outdata

        return new_ps

    def get_subset_maps(self,
                        seg_subset: npt.NDArray,
                        hru_noroute: npt.NDArray | None = None) -> tuple[dict[int, list[int]], dict[int, int]]:
        """Create dictionaries mapping segments to HRUs and HRUs to segments,
        filtered by a segment subset and optional non-routed HRUs.

        :param seg_subset: Array of segment IDs defining the spatial subset
        :param hru_noroute: Array of non-routed HRU IDs to include (segment=0)
        :returns: Tuple of (seg_to_hru, hru_to_seg) dictionaries
        """

        orig_hru_segment = self.get('hru_segment_nhm').data
        orig_nhm_id = self.get('nhm_id').data

        if hru_noroute is None:
            hru_noroute = np.array([], dtype=np.int32)

        seg_to_hru: dict[int, list[int]] = dict()
        hru_to_seg: dict[int, int] = dict()

        # Create boolean masks for efficient filtering
        in_seg_subset = np.isin(orig_hru_segment, seg_subset)
        in_hru_noroute = np.isin(orig_nhm_id, hru_noroute)

        # Process in original order to maintain ordering
        for cidx in range(len(orig_hru_segment)):
            cseg = orig_hru_segment[cidx]
            hid = orig_nhm_id[cidx]

            if in_seg_subset[cidx]:
                seg_to_hru.setdefault(cseg, []).append(hid)
                hru_to_seg[hid] = cseg
            elif in_hru_noroute[cidx]:
                if cseg != 0:
                    con.print(f'[orange3]WARNING[/]: Non-routed HRU {hid} routes to segment {cseg}; skipping.')
                else:
                    seg_to_hru.setdefault(cseg, []).append(hid)
                    hru_to_seg[hid] = cseg

        return seg_to_hru, hru_to_seg

    def get_output_order(self,
                         hru_to_seg: dict[int, int],
                         seg_to_hru: dict[int, list[int]],
                         new_nhm_seg_to_idx1: dict[int, int],
                         hru_noroute: npt.NDArray | None = None,
                         keep_hru_order: bool = False) -> tuple[list[int], list[int]]:
        """Create lists of HRU IDs and renumbered HRU segments for a model subset.

        HRU-related parameters can be output with either the legacy segment-oriented order
        or maintaining their original HRU-relative order from the parameter database.

        :param hru_to_seg: Dictionary mapping HRU IDs to segment IDs
        :param seg_to_hru: Dictionary mapping segment IDs to lists of HRU IDs
        :param new_nhm_seg_to_idx1: Dictionary mapping segment IDs to 1-based local indices
        :param hru_noroute: Array of non-routed HRU IDs to include
        :param keep_hru_order: If True, keep the original HRU-relative order
        :returns: Tuple of (hru_order_subset, new_hru_segment) lists
        """

        orig_hru_segment = self.get('hru_segment_nhm').data
        nhm_id_to_idx = self.get('nhm_id').index_map

        if hru_noroute is None:
            hru_noroute = np.array([], dtype=np.int32)

        if keep_hru_order:
            hru_order_subset = [kk for kk in hru_to_seg.keys()]

            new_hru_segment = []
            for cseg in hru_to_seg.values():
                if cseg in new_nhm_seg_to_idx1:
                    new_hru_segment.append(new_nhm_seg_to_idx1[cseg])
                elif cseg == 0:
                    new_hru_segment.append(0)
                else:
                    new_hru_segment.append(-1)
        else:
            # Get NHM HRU IDs ordered by the segments in the model subset
            hru_order_subset = []
            for cseg in new_nhm_seg_to_idx1.keys():
                if cseg in seg_to_hru:
                    for chru in seg_to_hru[cseg]:
                        hru_order_subset.append(chru)
                else:
                    con.print(f'[orange3]WARNING[/]: Stream segment {cseg} has no HRUs connected to it.')

            # Append the additional non-routed HRUs to the list
            if len(hru_noroute) > 0:
                for cseg in hru_noroute:
                    if orig_hru_segment[nhm_id_to_idx[cseg]] == 0:
                        hru_order_subset.append(cseg)
                    else:
                        con.print(f'[red]ERROR[/]: User-supplied HRU {cseg} routes to segment '
                                  f'{orig_hru_segment[nhm_id_to_idx[cseg]]}; skipping.')

            # Renumber hru_segments for the subset
            new_hru_segment = []
            for cseg, cidx in new_nhm_seg_to_idx1.items():
                if cseg in seg_to_hru:
                    for _ in seg_to_hru[cseg]:
                        new_hru_segment.append(cidx)

            # Append zeroes for each additional non-routed HRU
            if len(hru_noroute) > 0:
                for cseg in hru_noroute:
                    if orig_hru_segment[nhm_id_to_idx[cseg]] == 0:
                        new_hru_segment.append(0)

        return hru_order_subset, new_hru_segment

    def get_poi_subset(self,
                       new_nhm_seg_to_idx1: dict[int, int],
                       seg_to_hru: dict[int, list[int]],
                       addl_gages: dict[str, int] | None = None) -> tuple[list[int], list[str], list[int]]:
        """Create lists of POI segments, IDs, and types for a model subset.

        Subsets the poi_gage_segment, poi_gage_id, and poi_type parameters based on
        which segments are in the model subset. Optionally merges in user-specified
        additional streamgages with conflict resolution.

        :param new_nhm_seg_to_idx1: Dictionary mapping segment IDs to 1-based local indices
        :param seg_to_hru: Dictionary mapping segment IDs to lists of HRU IDs
        :param addl_gages: Optional dictionary of additional streamgages (gage_id -> nhm_seg)
        :returns: Tuple of (poi_gage_segment, poi_gage_id, poi_type) lists
        """

        new_poi_gage_segment: list[int] = []
        new_poi_gage_id: list[str] = []
        new_poi_type: list[int] = []

        if self.exists('poi_gage_segment'):
            poi_gage_segment = self.get('poi_gage_segment').tolist()
            poi_gage_id = self.get('poi_gage_id').tolist()
            poi_type = self.get('poi_type').tolist()

            # Look up each segment in the subset
            nhm_seg_dict = self.get('nhm_seg').index_map
            poi_gage_dict = self.get('poi_gage_segment').index_map

            for ss in new_nhm_seg_to_idx1.keys():
                sidx = nhm_seg_dict[ss] + 1
                if sidx in poi_gage_segment:
                    new_poi_gage_segment.append(new_nhm_seg_to_idx1[sidx])
                    new_poi_gage_id.append(poi_gage_id[poi_gage_dict[sidx]])
                    new_poi_type.append(poi_type[poi_gage_dict[sidx]])

            # Add any valid user-specified streamgage, nhm_seg pairs
            if addl_gages:
                for ss, vv in addl_gages.items():
                    if ss in new_poi_gage_id:
                        idx = new_poi_gage_id.index(ss)
                        con.print(f'[orange3]WARNING[/]: Existing POI {ss} overridden '
                                  f'(was {new_poi_gage_segment[idx]}, now {new_nhm_seg_to_idx1[vv]})')
                        new_poi_gage_segment[idx] = new_nhm_seg_to_idx1[vv]
                        new_poi_type[idx] = 0
                    elif new_nhm_seg_to_idx1[vv] in new_poi_gage_segment:
                        sidx = new_poi_gage_segment.index(new_nhm_seg_to_idx1[vv])
                        con.print(f'[orange3]WARNING[/]: User-specified streamgage ({ss}) '
                                  f'has same nhm_seg ({new_nhm_seg_to_idx1[vv]}) '
                                  f'as existing POI ({new_poi_gage_id[sidx]}); replacing streamgage ID')
                        new_poi_gage_id[sidx] = ss
                        new_poi_type[sidx] = 0
                    elif vv not in seg_to_hru.keys():
                        con.print(f'[orange3]WARNING[/]: User-specified streamgage ({ss}) has nhm_seg={vv} '
                                  f'which is not part of the model subset; skipping.')
                    else:
                        new_poi_gage_id.append(ss)
                        new_poi_gage_segment.append(new_nhm_seg_to_idx1[vv])
                        new_poi_type.append(0)

        return new_poi_gage_segment, new_poi_gage_id, new_poi_type

    def extract_subset(self,
                       dag_ds_subset: nx.DiGraph,
                       hru_noroute: npt.NDArray | None = None,
                       keep_hru_order: bool = False,
                       addl_gages: dict[str, int] | None = None) -> 'Parameters':
        """Extract a spatial subset of parameters given a stream network subgraph.

        This is a high-level method that orchestrates the full extraction workflow:
        derives segment lists from the DAG, builds HRU/segment mappings, determines
        HRU output order, subsets POIs, and creates the final Parameters subset.

        :param dag_ds_subset: Directed acyclic graph representing the stream network subset
            (as returned by get_streamnet_subset)
        :param hru_noroute: Array of non-routed HRU IDs to include (segment=0)
        :param keep_hru_order: If True, keep the original HRU-relative order
        :param addl_gages: Optional dictionary of additional streamgages (gage_id -> nhm_seg)
        :returns: New Parameters object containing the spatial subset

        :raises ValueError: If no routed HRUs are associated with the subset segments
        """

        if hru_noroute is None:
            hru_noroute = np.array([], dtype=np.int32)

        # Derive segment arrays from the DAG edges
        new_nhm_seg = np.array([ee[0] for ee in dag_ds_subset.edges])

        # Build 1-based index mapping for segments
        new_nhm_seg_to_idx1 = dict((ss, ii + 1) for ii, ss in enumerate(new_nhm_seg))

        # Generate renumbered local tosegments (1-based, zero = outlet)
        new_tosegment = [new_nhm_seg_to_idx1[ee[1]] if ee[1] in new_nhm_seg_to_idx1
                         else 0 for ee in dag_ds_subset.edges]

        # Build filtered seg-to-hru and hru-to-seg mappings
        seg_to_hru, hru_to_seg = self.get_subset_maps(new_nhm_seg, hru_noroute)

        if set(hru_to_seg.values()) == set(hru_noroute):
            raise ValueError('No routed HRUs are associated with any of the subset segments')

        # Determine HRU output order and renumbered hru_segment
        hru_order_subset, new_hru_segment = self.get_output_order(hru_to_seg, seg_to_hru,
                                                                  new_nhm_seg_to_idx1, hru_noroute,
                                                                  keep_hru_order=keep_hru_order)

        # Subset POI parameters
        new_poi_gage_segment, new_poi_gage_id, new_poi_type = self.get_poi_subset(new_nhm_seg_to_idx1,
                                                                                  seg_to_hru,
                                                                                  addl_gages=addl_gages)

        # Create the parameter subset
        return self.create_subset(hru_order_subset, new_hru_segment, new_nhm_seg,
                                  new_poi_gage_id, new_poi_gage_segment,
                                  new_poi_type, new_tosegment)
