#!/usr/bin/env python

import argparse
# import datetime
import glob
# import netCDF4
import os

import io
import pkgutil
# import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

# import write_output_var as wr

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Convert parameter files to different formats')
    parser.add_argument('--src', help='Source model output directory')
    parser.add_argument('--dst', help='Destination file or directory')
    # parser.add_argument('-v', '--var', help='Variable to process')

    args = parser.parse_args()

    xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/variables.xml').decode('utf-8'))
    xml_tree = xmlET.parse(xml_fh)
    xml_root = xml_tree.getroot()

    datatype_map = {'I': 4, 'F': 5, 'D': 6, 'S': 2}
    # Datatypes: NC_FLOAT=5, NC_DOUBLE=6, NC_INT=4, NC_CHAR=2

    # Map PRMS dimension names to netcdf dimension names
    dims_map = {'nhru': 'hru', 'nsegment': 'segment'}

    var_dict = {}

    # Iterate over child nodes of root
    for elem in xml_root.findall('variable'):
        # print(elem.attrib.get('name'))
        name = elem.attrib.get('name')
        dtype = elem.find('type').text
        desc = elem.find('desc').text
        units = elem.find('units').text

        # Add dimensions for current parameter
        dims_list = []
        for cdim in elem.findall('./dimensions/dimension'):
            dims_list.append(cdim.attrib.get('name'))

        dims = ','.join(dims_list)
        # Print the elements of the node
        # print(elem.find('desc').text)
        # print(elem.find('size').text)
        # dim_size = int(elem.find('size').text)
        var_dict[name] = [dtype, desc, units, dims]

    # variables = list(var_dict.keys())

    filelist = glob.glob('{}/*.csv'.format(args.src))

    for curr_file in filelist:
        filename = os.path.basename(curr_file)
        dim_name = filename.split('_')[0]
        var_name = '_'.join(os.path.splitext(filename)[0].split('_')[1:])

        print('-' * 40)
        try:
            nc_dim_name = dims_map[dim_name]
            var_type = var_dict[var_name][0]
            var_desc = var_dict[var_name][1]
            var_units = var_dict[var_name][2]

            print('Calling fortran routine')
            print('Source file: {}'.format(curr_file))
            print('Variable name: {}'.format(var_name))
            print('Variable type: {}'.format(var_type))
            print('Variable description: {}'.format(var_desc))
            print('Variable units: {}'.format(var_units))
            print('Variable dimension: {}'.format(dim_name))
            print('netCDF dimension name: {}'.format(nc_dim_name))
            print('Write output to: {}.nc'.format(var_name))

            #     wr.write_output_var(src=curr_file, var_name='tmaxf', var_desc='Maximum temperature',
            #                         var_type=6, var_units='degrees_fahrenheit', var_dim='hru',
            #                         ntime=13241, nsize=109951)
        except KeyError:
            print('WARNING: {} is not a valid output variable or has an invalid dimension; skipping.'.format(filename))

    # print('.done!')


if __name__ == '__main__':
    main()
