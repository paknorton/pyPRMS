
from __future__ import (absolute_import, division, print_function)

import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import CATEGORY_DELIM, DIMENSIONS_HDR, PARAMETERS_HDR, VAR_DELIM
# from pyPRMS.constants import PARAMETERS_XML, DIMENSIONS_XML

import functools


class ParameterFile(ParameterSet):
    def __init__(self, filename):
        super(ParameterFile, self).__init__()

        self.__filename = None
        self.__header = None

        self.__isloaded = False
        self.filename = filename

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, name):
        self.__isloaded = False
        self.__filename = name
        self.__header = []  # Initialize the list of file headers

        self._read()

    @property
    def headers(self):
        """Returns the headers read from the parameter file"""
        return self.__header

    def _read(self):
        # Read the parameter file into memory and parse it
        infile = open(self.filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Grab the header stuff first
        for line in it:
            if line.strip('* ') == DIMENSIONS_HDR:
                break
            self.__header.append(line)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Now process the dimensions
        for line in it:
            if line.strip('* ') == PARAMETERS_HDR:
                break
            if line == VAR_DELIM:
                continue

            # Add dimension - all dimensions are scalars
            self.dimensions.add(line, int(next(it)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Lastly process the parameters
        for line in it:
            if line == VAR_DELIM:
                continue
            varname = line.split(' ')[0]

            try:
                self.parameters.add(varname)
            except ParameterError:
                print('%s: Duplicate parameter name.. skipping' % varname)

                # Skip to the next parameter
                try:
                    while next(it) != VAR_DELIM:
                        pass
                except StopIteration:
                    # Hit end of file
                    pass
                continue

            # Read in the dimension names
            ndims = int(next(it))  # number of dimensions for this variable
            dim_tmp = [next(it) for _ in range(ndims)]

            # Lookup dimension size for each dimension name
            arr_shp = [self.dimensions.get(dd).size for dd in dim_tmp]

            # Compute the total size of the parameter
            dim_size = functools.reduce(lambda x, y: x * y, arr_shp)

            # Total dimension size declared for parameter in file; it should be total size of declared dimensions.
            numval = int(next(it))

            self.parameters.get(varname).datatype = int(next(it))

            # Add the dimensions to the parameter, dimension size is looked up from the global Dimensions object
            for dd in dim_tmp:
                self.parameters.get(varname).dimensions.add(dd, self.dimensions.get(dd).size)

            if numval != dim_size:
                # The declared total size doesn't match the total size of the declared dimensions
                print('{}: Declared total size for parameter does not match the total size of the ' +
                      'declared dimension(s) ({} != {}).. skipping'.format(varname, numval, dim_size))

                # Still have to read all the values to skip this properly
                try:
                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                except StopIteration:
                    # Hit the end of the file
                    pass
                self.parameters.del_param(varname)
            else:
                # Check if number of values written match the number of values declared
                try:
                    # Read in the data values
                    vals = []

                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                        vals.append(cval)
                except StopIteration:
                    # Hit the end of the file
                    pass

                if len(vals) != numval:
                    print('{}: number of values does not match dimension size ' +
                          '({} != {}).. skipping'.format(varname, len(vals), numval))

                    # Remove the parameter from the dictionary
                    self.parameters.del_param(varname)
                else:
                    # Convert the values to the correct datatype
                    self.parameters.get(varname).data = vals

        self.__isloaded = True

    def write(self, filename):
        """Write a parameter file using defined dimensions and parameters"""

        # Write the parameters out to a file
        outfile = open(filename, 'w')

        for hh in self.__header:
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
