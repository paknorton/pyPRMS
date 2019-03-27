
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import netCDF4 as nc
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions
from pyPRMS.constants import NETCDF_DATATYPES, NHM_DATATYPES, PARAMETERS_XML, DIMENSIONS_XML
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
            print(vv.name, curr_datatype)
            sys.stdout.flush()

            if curr_datatype != 'S1':
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
                    if not isinstance(vv.minimum, basestring):
                        curr_param.valid_min = vv.minimum

                if vv.maximum is not None:
                    if not isinstance(vv.maximum, basestring):
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
