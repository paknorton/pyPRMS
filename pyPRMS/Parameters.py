
# from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

import numpy as np
import pandas as pd
from collections import OrderedDict
# import xml.etree.ElementTree as xmlET

from pyPRMS.Parameter import Parameter
from pyPRMS.Exceptions_custom import ParameterError
# from pyPRMS.constants import DATA_TYPES
# from pyPRMS.Dimensions import ParamDimensions


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

    @property
    def parameters(self):
        """Returns an ordered dictionary of parameter objects.

        :rtype: collections.OrderedDict[str, Parameter]
        """

        return self.__parameters

    def add(self, name, datatype=None, units=None, model=None, description=None,
            help=None, modules=None, minimum=None, maximum=None, default=None,
            info=None):
        """Add a new parameter by name.

        :param str name: A valid PRMS parameter name
        :param int datatype: The datatype for the parameter (1-Integer, 2-Float, 3-Double, 4-String)
        :param str units: Option units string for the parameter
        :param str model: <<FILL IN LATER>>
        :param str description: Description of the parameter
        :param str help: Help text for the parameter
        :param modules: List of modules that require the parameter
        :type modules: list[str] or None
        :param minimum: Minimum value allowed in the parameter data
        :type minimum: int or float or None
        :param maximum: Maximum value allowed in the parameter data
        :type maximum: int or float or None
        :param default: Default value used for parameter data
        :type default: int or float or None
        :param info: Parameter object containing the metadata information for the parameter
        :type info: Parameter

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

            print(pp.check())

            if not pp.check_values():
                print('    WARNING: Value(s) (range: {}, {}) outside the valid range of ({}, {})'.format(pp.data.min(), pp.data.max(), pp.minimum, pp.maximum))

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

    def remove(self, name):
        """Delete one or more parameters if they exist.

        :param name: parameter or list of parameters to remove
        :type name: str or list[str]
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

    def exists(self, name):
        """Checks if a given parameter name exists.

        :param str name: Name of the parameter
        :returns: True if parameter exists, otherwise False
        :rtype: bool
        """

        return name in self.parameters.keys()

    def get(self, name):
        """Returns the given parameter object.

        :param str name: The name of the parameter
        :returns: Parameter object
        :rtype: Parameter
        """

        # Return the given parameter
        if self.exists(name):
            return self.__parameters[name]
        raise ValueError('Parameter, {}, does not exist.'.format(name))

    def get_dataframe(self, name):
        """Returns a pandas DataFrame for a parameter.

        If the parameter dimensions includes either nhrus or nsegment then the
        respective national ids are included, if they exist, as the index in the
        returned dataframe.

        :param str name: The name of the parameter
        :returns: Pandas DataFrame of the parameter data
        :rtype: pd.DataFrame
        """

        if self.exists(name):
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
                param_id = self.__parameters['nhm_seg'].as_dataframe

                # Create a DataFrame of the parameter
                param_data = param_data.merge(param_id, left_index=True, right_index=True)
                param_data.set_index(['nhm_seg'], inplace=True)
            elif name == 'snarea_curve':
                # Special handling for snarea_curve parameter
                param_data = pd.DataFrame(cparam.as_dataframe.values.reshape((-1, 11)))
                param_data.rename(columns={k: k+1 for k in param_data.columns},
                                  index={k: k+1 for k in param_data.index},
                                  inplace=True)
                param_data.index.name = 'curve_index'
            return param_data
        raise ValueError('Parameter, {}, has no associated data'.format(name))

    def get_subset(self, name, global_ids):
        """Returns a subset for a parameter based on the global_ids (e.g. nhm)"""
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

    def remove_by_global_id(self, hrus=None, segs=None):
        """Removes data-by-id (nhm_seg, nhm_id) from all parameters"""

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
    #             arr_shp = [self.__paramdict['Dimensions'][dd] for dd in thevar['dimnames']]
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
