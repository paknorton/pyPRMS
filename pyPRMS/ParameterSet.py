
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import netCDF4 as nc
import os
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions
from pyPRMS.constants import CATEGORY_DELIM, NETCDF_DATATYPES, NHM_DATATYPES, PARAMETERS_XML, DIMENSIONS_XML, VAR_DELIM
from pyPRMS.prms_helpers import float_to_str


class ParameterSet(object):
    """
    A parameteter set which is a container for a Parameters objects and a Dimensions objects.
    """

    def __init__(self):
        """Create a new ParameterSet"""

        self.__parameters = Parameters()
        self.__dimensions = Dimensions()

    @property
    def dimensions(self):
        """Returns the Dimensions object"""
        return self.__dimensions

    @property
    def parameters(self):
        """Returns the Parameters object"""
        return self.__parameters

    @property
    def xml_global_dimensions(self):
        """Return an xml ElementTree of the dimensions used by all parameters"""

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
        """Return an xml ElementTree of the parameters"""
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
                if isinstance(vv.minimum, basestring):
                    xmlET.SubElement(param_sub, 'minimum').text = vv.minimum
                else:
                    xmlET.SubElement(param_sub, 'minimum').text = float_to_str(vv.minimum)
            if vv.maximum is not None:
                if isinstance(vv.maximum, basestring):
                    xmlET.SubElement(param_sub, 'maximum').text = vv.maximum
                else:
                    xmlET.SubElement(param_sub, 'maximum').text = float_to_str(vv.maximum)
            if vv.default is not None:
                if isinstance(vv.default, basestring):
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
        assert False, 'ParameterSet._read() must be defined by child class'

    def remove_unneeded_parameters(self, required_params=None):
        """Given a set of required parameters, removes parameters that are not
        needed"""

        if isinstance(required_params, set):
            remove_list = set(self.parameters.keys()).difference(required_params)
        elif isinstance(required_params, list):
            remove_list = set(self.parameters.keys()).difference(set(required_params))
        else:
            raise TypeError('remove_unneeded_parameters() requires a set or list argument')

        for rparam in remove_list:
            self.parameters.remove(rparam)

    def write_parameters_xml(self, output_dir):
        """Write global parameters.xml"""
        # Write the global parameters xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_parameters)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, PARAMETERS_XML), 'w') as ff:
            ff.write(xmlstr)

    def write_dimensions_xml(self, output_dir):
        """Write global dimensions.xml"""
        # Write the global dimensions xml file
        xmlstr = minidom.parseString(xmlET.tostring(self.xml_global_dimensions)).toprettyxml(indent='    ')
        with open('{}/{}'.format(output_dir, DIMENSIONS_XML), 'w') as ff:
            ff.write(xmlstr)

    def write_netcdf(self, filename):
        """Write parameters to a netcdf file"""

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
                        curr_param = nc_hdl.createVariable(vv.name, curr_datatype, tuple(list(vv.dimensions.keys())[::-1]),
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
                str_size = len(max(vv.data, key=len))

                # Create a dimension for the string length
                nc_hdl.createDimension(vv.name + '_nchars', str_size)

                # Temporary to add extra dimension for number of characters
                tmp_dims = vv.dimensions.keys()
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

                # Write the data
                if len(tmp_dims) == 1:
                    curr_param[:] = nc.stringtochar(vv.data)
                elif len(tmp_dims) == 2:
                    curr_param[:, :] = nc.stringtochar(vv.data)
            sys.stdout.flush()
        # Close the netcdf file
        nc_hdl.close()

    def write_paramdb(self, output_dir):
        """Write all parameters using the paramDb output format"""

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
                ff.write(xmlstr.encode('utf-8'))

    def write_parameter_file(self, filename, header=None):
        """Write a parameter file using defined dimensions and parameters"""

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
                    for xx in vv.data.flatten(order='A'):
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
