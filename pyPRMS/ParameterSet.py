
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import netCDF4 as nc
import numpy as np
import os
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions
from pyPRMS.ValidParams import ValidParams
from pyPRMS.constants import CATEGORY_DELIM, NETCDF_DATATYPES, NHM_DATATYPES, PARAMETERS_XML
from pyPRMS.constants import DIMENSIONS_XML, VAR_DELIM, HRU_DIMS
from pyPRMS.prms_helpers import float_to_str


class ParameterSet(object):

    """
    Container for a Parameters object and a Dimensions object.
    """

    def __init__(self, verbose=False, verify=True):
        """Create a new ParameterSet.

        :param bool verbose: output debugging information
        :param bool verify: whether to load the master parameters (default=True)
        """

        self.__parameters = Parameters()
        self.__dimensions = Dimensions()

        self.__master_params = None
        if verify:
            self.__master_params = ValidParams()

        self.verbose = verbose

    @property
    def dimensions(self):
        """Get dimensions object.

        :returns: dimensions object
        :rtype: Dimensions
        """

        return self.__dimensions

    @property
    def master_parameters(self):
        """Get master parameters.

        :returns: master parameters object
        :rtype: ValidParams
        """

        return self.__master_params

    @property
    def parameters(self):
        """Get Parameters object.

        :returns: Parameters object
        :rtype: Parameters
        """

        return self.__parameters

    @property
    def xml_global_dimensions(self):
        """Get XML element tree of the dimensions used by all parameters.

        :returns: element tree of dimensions
        :rtype: xmlET.Element
        """

        dims_xml = xmlET.Element('dimensions')

        for kk, vv in iteritems(self.dimensions):
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)

            if vv.description:
                xmlET.SubElement(dim_sub, 'desc').text = vv.description

            xmlET.SubElement(dim_sub, 'size').text = float_to_str(vv.size)

        return dims_xml

    @property
    def xml_global_parameters(self):
        """Get XML element tree of the parameters.

        :returns: element tree of parameters
        :rtype: xmlET.Element
        """

        inv_map = {vv: kk for kk, vv in iteritems(NHM_DATATYPES)}
        # print(inv_map)

        params_xml = xmlET.Element('parameters')

        for vv in self.parameters.values():
            # print(vv.name, inv_map[vv.datatype])
            param_sub = xmlET.SubElement(params_xml, 'parameter')
            param_sub.set('name', vv.name)

            xmlET.SubElement(param_sub, 'type').text = inv_map[vv.datatype]

            if vv.units:
                xmlET.SubElement(param_sub, 'units').text = vv.units
            if vv.model:
                xmlET.SubElement(param_sub, 'model').text = vv.model
            if vv.description:
                xmlET.SubElement(param_sub, 'desc').text = vv.description
            if vv.help:
                xmlET.SubElement(param_sub, 'help').text = vv.help
            if vv.minimum is not None:
                if isinstance(vv.minimum, str):
                    xmlET.SubElement(param_sub, 'minimum').text = vv.minimum
                else:
                    xmlET.SubElement(param_sub, 'minimum').text = float_to_str(vv.minimum)
            if vv.maximum is not None:
                if isinstance(vv.maximum, str):
                    xmlET.SubElement(param_sub, 'maximum').text = vv.maximum
                else:
                    xmlET.SubElement(param_sub, 'maximum').text = float_to_str(vv.maximum)
            if vv.default is not None:
                if isinstance(vv.default, str):
                    xmlET.SubElement(param_sub, 'default').text = vv.default
                else:
                    xmlET.SubElement(param_sub, 'default').text = float_to_str(vv.default)

            if bool(vv.modules):
                modules_sub = xmlET.SubElement(param_sub, 'modules')

                for mm in vv.modules:
                    xmlET.SubElement(modules_sub, 'module').text = mm

            param_sub.append(vv.dimensions.xml)

        return params_xml

    def _read(self):
        """Abstract function for reading.
        """

        assert False, 'ParameterSet._read() must be defined by child class'

    def degenerate_parameters(self):
        """List parameters that have fewer dimensions than specified in the master parameters."""

        if self.__master_params is not None:
            for kk, vv in iteritems(self.parameters):
                try:
                    if set(vv.dimensions.keys()) != set(self.__master_params[kk].dimensions.keys()):
                        if not (set(self.__master_params[kk].dimensions.keys()).issubset(set(HRU_DIMS)) and
                                set(vv.dimensions.keys()).issubset(HRU_DIMS)):
                            print('Parameter, {}, is degenerate'.format(kk))
                            print('  parameter: ', list(vv.dimensions.keys()))
                            print('     master: ', list(self.__master_params[kk].dimensions.keys()))
                except ValueError:
                    print('ERROR: Parameter, {}, is not a valid PRMS parameter'.format(kk))

    def expand_parameter(self, name):
        """Expand an existing parameter.

        Expands (e.g. reshape) a parameter, broadcasting existing value(s) into
        new shape specified by master parameters. The hru_deplcrv parameter has
        special handling to also update the snarea_curve parameter.

        :param str name: name of parameter
        """

        if self.__master_params is not None:
            # 1) make sure parameter exists
            if self.__master_params.exists(name):
                # 2) get dimensions from master parameters
                new_dims = self.__master_params.parameters[name].dimensions.copy()

                # The new_dims copy is no longer of type Dimensions, instead it
                # is an OrderedDict
                # 3) get dimension sizes from global dimensions object
                for kk, vv in iteritems(new_dims):
                    vv.size = self.__dimensions[kk].size

                if set(new_dims.keys()) == set(self.__parameters[name].dimensions.keys()):
                    print('Parameter, {}, already has the maximum number of dimensions'.format(name))
                    print('    current: ', list(self.__parameters[name].dimensions.keys()))
                    print('  requested: ', list(new_dims.keys()))
                else:
                    # 4) call reshape for the parameter
                    self.__parameters[name].reshape(new_dims)

                    if name == 'hru_deplcrv':
                        # hru_deplcrv needs special handling
                        # 2) get current value of hru_deplcrv, this is the snow_index to use
                        # 3) replace broadcast original value with np.arange(1:nhru)
                        orig_index = self.__parameters[name].data[0] - 1
                        new_indices = np.arange(1, new_dims['nhru'].size + 1)
                        self.__parameters['hru_deplcrv'].data = new_indices

                        # 5) get snarea_curve associated with original hru_deplcrv value
                        curr_snarea_curve = self.__parameters['snarea_curve'].data.reshape((-1, 11))[orig_index, :]

                        # 6) replace current snarea_curve values with broadcast of select snarea_curve*nhru
                        new_snarea_curve = np.broadcast_to(curr_snarea_curve, (new_dims['nhru'].size, 11))
                        # 7) reset snarea_curve dimension size to nhru*11
                        self.__parameters['snarea_curve'].dimensions['ndeplval'].size = new_dims['nhru'].size * 11
                        self.__parameters['snarea_curve'].data = new_snarea_curve.flatten(order='C')

                        if self.verbose:
                            print('hru_deplcrv and snarea_curve have been expanded/updated')

    def reduce_by_modules(self, control=None):
        """Reduce the ParameterSet to the parameters required by the modules
        defined in a control file.
        """

        if self.__master_params is not None:
            pset = self.master_parameters.get_params_for_modules(modules=control.modules.values())
            self.reduce_parameters(required_params=pset)

    def reduce_parameters(self, required_params=None):
        """Remove parameters that are not needed.

        Given a set of required parameters removes parameters that are not
        listed.

        :param required_params: list or set of required parameters names
        :type required_params: list or set

        :raises TypeError: if required_params is not a set or list
        """

        if isinstance(required_params, set):
            remove_list = set(self.parameters.keys()).difference(required_params)
        elif isinstance(required_params, list):
            remove_list = set(self.parameters.keys()).difference(set(required_params))
        else:
            raise TypeError('remove_unneeded_parameters() requires a set or list argument')

        for rparam in remove_list:
            self.parameters.remove(rparam)

    def remove_by_global_id(self, hrus=None, segs=None):
        """Removes data-by-id (nhm_seg, nhm_id) from all parameters"""
        self.__parameters.remove_by_global_id(hrus=hrus, segs=segs)

        # Adjust the global dimensions
        if segs is not None:
            self.__dimensions['nsegment'].size -= len(segs)

        if hrus is not None:
            self.__dimensions['nhru'].size -= len(hrus)

            if self.__dimensions.exists('nssr'):
                self.__dimensions['nssr'].size -= len(hrus)
            if self.__dimensions.exists('ngw'):
                self.__dimensions['ngw'].size -= len(hrus)

    def write_parameters_xml(self, output_dir):
        """Write global parameters.xml file.

        :param str output_dir: output path for parameters.xml file
        """

        # Write the global parameters xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_parameters)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, PARAMETERS_XML), 'w') as ff:
            ff.write(xmlstr)

    def write_dimensions_xml(self, output_dir):
        """Write global dimensions.xml file.

        :param str output_dir: output path for dimensions.xml file
        """

        # Write the global dimensions xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_dimensions)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, DIMENSIONS_XML), 'w') as ff:
            ff.write(xmlstr)

    def write_netcdf(self, filename):
        """Write parameters to a netcdf format file.

        :param str filename: full path for output file
        """

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
            curr_datatype = NETCDF_DATATYPES[vv.datatype]
            # print(vv.name, curr_datatype)
            sys.stdout.flush()

            if curr_datatype != 'S1':
                try:
                    if vv.dimensions.keys()[0] == 'one':
                        # Scalar values
                        curr_param = nc_hdl.createVariable(vv.name, curr_datatype,
                                                           fill_value=nc.default_fillvals[curr_datatype], zlib=True)
                    else:
                        # The variable dimensions are stored with C-ordering (slowest -> fastest)
                        # The variables in this library are based on Fortran-ordering (fastest -> slowest)
                        # so we reverse the order of the dimensions and the arrays for
                        # writing out to the netcdf file.
                        # dtmp = vv.dimensions.keys()
                        # dtmp.reverse()
                        curr_param = nc_hdl.createVariable(vv.name, curr_datatype, tuple(vv.dimensions.keys()[::-1]),
                                                           fill_value=nc.default_fillvals[curr_datatype], zlib=True)
                        # curr_param = nc_hdl.createVariable(vv.name, curr_datatype, tuple(vv.dimensions.keys()),
                        #                                    fill_value=nc.default_fillvals[curr_datatype], zlib=True)
                except TypeError:
                    # python 3.x
                    if list(vv.dimensions.keys())[0] == 'one':
                        # Scalar values
                        curr_param = nc_hdl.createVariable(vv.name, curr_datatype,
                                                           fill_value=nc.default_fillvals[curr_datatype], zlib=True)
                    else:
                        curr_param = nc_hdl.createVariable(vv.name, curr_datatype,
                                                           tuple(list(vv.dimensions.keys())[::-1]),
                                                           fill_value=nc.default_fillvals[curr_datatype], zlib=True)

                # Add the attributes
                if vv.help:
                    curr_param.description = vv.help
                elif vv.description:
                    # Try to fallback to the description if no help text
                    curr_param.description = vv.description

                if vv.units:
                    curr_param.units = vv.units

                # NOTE: Sometimes a warning is output from the netcdf4 library
                #       warning that valid_min and/or valid_max
                #       cannot be safely cast to variable dtype.
                #       Not sure yet what is causing this.
                if vv.minimum is not None:
                    # TODO: figure out how to handle bounded parameters
                    if not isinstance(vv.minimum, str):
                        curr_param.valid_min = vv.minimum

                if vv.maximum is not None:
                    if not isinstance(vv.maximum, str):
                        curr_param.valid_max = vv.maximum

                # Write the data
                if len(vv.dimensions.keys()) == 1:
                    curr_param[:] = vv.data
                elif len(vv.dimensions.keys()) == 2:
                    curr_param[:, :] = vv.data.transpose()
            else:
                # String parameter
                # Get the maximum string length in the array of data
                # print('String parameter: {}'.format(vv.name))
                str_size = len(max(vv.data, key=len))
                # print('size: {}'.format(str_size))

                # Create a dimension for the string length
                nc_hdl.createDimension(vv.name + '_nchars', str_size)

                # Temporary to add extra dimension for number of characters
                tmp_dims = list(vv.dimensions.keys())
                tmp_dims.extend([vv.name + '_nchars'])
                curr_param = nc_hdl.createVariable(vv.name, curr_datatype, tuple(tmp_dims),
                                                   fill_value=nc.default_fillvals[curr_datatype], zlib=True)

                # Add the attributes
                if vv.help:
                    curr_param.description = vv.help
                elif vv.description:
                    # Try to fallback to the description if no help text
                    curr_param.description = vv.description

                if vv.units:
                    curr_param.units = vv.units

                # print('dimensions: {}'.format(tmp_dims))
                # print('original data shape: ', vv.data.shape)
                # print('original data type: ', vv.data.dtype)
                # print('data:')
                # print(nc.stringtochar(vv.data, encoding='none'))
                # print('data shape: ', nc.stringtochar(vv.data, encoding='none').shape)
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

    def write_paramdb(self, output_dir):
        """Write all parameters using the paramDb output format.

        :param str output_dir: output path for paramDb files
        """

        # check for / create output directory
        try:
            print('Creating output directory: {}'.format(output_dir))
            os.makedirs(output_dir)
        except OSError:
            print("\tUsing existing directory")

        # Write the global dimensions xml file
        self.write_dimensions_xml(output_dir)
        # xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_dimensions)).toprettyxml(indent='    ')
        # with open('{}/{}'.format(output_dir, DIMENSIONS_XML), 'w') as ff:
        #     ff.write(xmlstr)

        # Write the global parameters xml file
        self.write_parameters_xml(output_dir)
        # xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_parameters)).toprettyxml(indent='    ')
        # with open('{}/{}'.format(output_dir, PARAMETERS_XML), 'w') as ff:
        #     ff.write(xmlstr)

        for xx in self.parameters.values():
            # Write out each parameter in the paramDb csv format
            with open('{}/{}.csv'.format(output_dir, xx.name), 'w') as ff:
                ff.write(xx.toparamdb())

            # Write xml file for the parameter
            xmlstr = minidom.parseString(xmlET.tostring(xx.xml)).toprettyxml(indent='    ')
            with open('{}/{}.xml'.format(output_dir, xx.name), 'w') as ff:
                # ff.write(xmlstr.encode('utf-8'))
                ff.write(xmlstr)

    def write_parameter_file(self, filename, header=None):
        """Write a parameter file.

        :param str filename: name of parameter file
        :param list[str] header: list of header lines
        """

        # Write the parameters out to a file
        outfile = open(filename, 'w')

        if header:
            for hh in header:
                # Write out any header stuff
                outfile.write('{}\n'.format(hh))

        # Dimension section must be written first
        outfile.write('{} Dimensions {}\n'.format(CATEGORY_DELIM, CATEGORY_DELIM))

        for (kk, vv) in self.dimensions.items():
            # Write each dimension name and size separated by VAR_DELIM
            outfile.write('{}\n'.format(VAR_DELIM))
            outfile.write('{}\n'.format(kk))
            outfile.write('{:d}\n'.format(vv.size))

        # Now write out the Parameter category
        order = ['name', 'dimensions', 'datatype', 'data']

        outfile.write('{} Parameters {}\n'.format(CATEGORY_DELIM, CATEGORY_DELIM))

        for vv in self.parameters.values():
            datatype = vv.datatype

            for item in order:
                # Write each variable out separated by self.__rowdelim
                if item == 'dimensions':
                    # Write number of dimensions first
                    outfile.write('{}\n'.format(vv.dimensions.ndims))

                    for dd in vv.dimensions.values():
                        # Write dimension names
                        outfile.write('{}\n'.format(dd.name))
                elif item == 'datatype':
                    # dimsize (which is computed) must be written before datatype
                    outfile.write('{}\n'.format(vv.data.size))
                    outfile.write('{}\n'.format(datatype))
                elif item == 'data':
                    # Write one value per line
                    # WARNING: 2019-10-10: had to change next line from order='A' to order='F'
                    #          because flatten with 'A' was only honoring the Fortran memory layout
                    #          if the array was contiguous which isn't always the
                    #          case if the arrays have been altered in size.
                    for xx in vv.data.flatten(order='F'):
                        if datatype in [2, 3]:
                            # Float and double types have to be formatted specially so
                            # they aren't written in exponential notation or with
                            # extraneous zeroes
                            tmp = '{:<20f}'.format(xx).rstrip('0 ')
                            if tmp[-1] == '.':
                                tmp += '0'

                            outfile.write('{}\n'.format(tmp))
                        else:
                            outfile.write('{}\n'.format(xx))
                elif item == 'name':
                    # Write the self.__rowdelim before the variable name
                    outfile.write('{}\n'.format(VAR_DELIM))
                    outfile.write('{}\n'.format(vv.name))

        outfile.close()
