
# class ParameterSet(object):
    # def degenerate_parameters(self):
    #     """Print parameters that have fewer dimensions than specified in the master parameters."""
    #
    #     result = []
    #     # TODO: 20230726 PAN - This is not needed with the current parameter code
    #     if self.__master_params is not None:
    #         for kk, vv in self.parameters.items():
    #             try:
    #                 if set(vv.dimensions.keys()) != set(self.__master_params[kk].dimensions.keys()):
    #                     if not (set(self.__master_params[kk].dimensions.keys()).issubset(set(HRU_DIMS)) and
    #                             set(vv.dimensions.keys()).issubset(HRU_DIMS)):
    #                         result.append(kk)
    #                         if self.verbose:
    #                             print(f'Parameter, {kk}, is degenerate')
    #                             print('  parameter: ', list(vv.dimensions.keys()))
    #                             print('     master: ', list(self.__master_params[kk].dimensions.keys()))
    #             except ValueError:
    #                 if self.verbose:
    #                     print(f'ERROR: Parameter, {kk}, is not a valid PRMS parameter')
    #     return result

    # def expand_parameter(self, name: str):
    #     """Expand an existing parameter.
    #
    #     Expand (e.g. reshape) a parameter, broadcasting existing value(s) into
    #     new shape specified by master parameters. The hru_deplcrv parameter has
    #     special handling to also update the snarea_curve parameter.
    #
    #     :param name: name of parameter
    #     """
    #     # TODO: 20230726 PAN - This is not needed with the current parameter code
    #     if self.__master_params is not None:
    #         # 1) make sure parameter exists
    #         if self.__master_params.exists(name):
    #             # 2) get dimensions from master parameters
    #             new_dims = self.__master_params.parameters[name].dimensions.copy()
    #
    #             # The new_dims copy is no longer of type Dimensions, instead it
    #             # is an OrderedDict
    #             # 3) get dimension sizes from global dimensions object
    #             for kk, vv in new_dims.items():
    #                 vv.size = self.__dimensions[kk].size
    #
    #             if self.verbose and set(new_dims.keys()) == set(self.__parameters[name].dimensions.keys()):
    #                 print(f'Parameter, {name}, already has the maximum number of dimensions')
    #                 print('    current: ', list(self.__parameters[name].dimensions.keys()))
    #                 print('  requested: ', list(new_dims.keys()))
    #
    #                 # TODO: Write special case where hru_deplcrv is dimensioned nhru, but
    #                 #       the number of snarea_curve entries is less than nhru * 11.
    #             else:
    #                 # 4) call reshape for the parameter
    #                 self.__parameters[name].reshape(new_dims)
    #
    #                 if name == 'hru_deplcrv':
    #                     # hru_deplcrv needs special handling
    #                     # 2) get current value of hru_deplcrv, this is the snow_index to use
    #                     # 3) replace broadcast original value with np.arange(1:nhru)
    #                     orig_index = self.__parameters[name].data[0] - 1
    #                     new_indices = np.arange(1, new_dims['nhru'].size + 1)
    #                     self.__parameters['hru_deplcrv'].data = new_indices
    #
    #                     # 5) get snarea_curve associated with original hru_deplcrv value
    #                     curr_snarea_curve = self.__parameters['snarea_curve'].data.reshape((-1, 11))[orig_index, :]
    #
    #                     # 6) replace current snarea_curve values with broadcast of select snarea_curve*nhru
    #                     new_snarea_curve = np.broadcast_to(curr_snarea_curve, (new_dims['nhru'].size, 11))
    #                     # 7) reset snarea_curve dimension size to nhru*11
    #                     self.__parameters['snarea_curve'].dimensions['ndeplval'].size = new_dims['nhru'].size * 11
    #                     self.__parameters['snarea_curve'].data = new_snarea_curve.flatten(order='C')
    #
    #                     if self.verbose:
    #                         print('hru_deplcrv and snarea_curve have been expanded/updated')

    # def remove_by_global_id(self, hrus: Optional[List] = None,
    #                         segs: Optional[List] = None):
    #     """Removes data-by-id (nhm_seg, nhm_id) from all parameters.
    #
    #     :param hrus: List of HRU IDs to remove
    #     :param segs: List of segment IDs to remove
    #     """
    #     self.__parameters.remove_by_global_id(hrus=hrus, segs=segs)
    #
    #     # Adjust the global dimensions
    #     if segs is not None:
    #         self.__dimensions['nsegment'].size -= len(segs)
    #
    #     if hrus is not None:
    #         self.__dimensions['nhru'].size -= len(hrus)
    #
    #         if self.__dimensions.exists('nssr'):
    #             self.__dimensions['nssr'].size -= len(hrus)
    #         if self.__dimensions.exists('ngw'):
    #             self.__dimensions['ngw'].size -= len(hrus)
